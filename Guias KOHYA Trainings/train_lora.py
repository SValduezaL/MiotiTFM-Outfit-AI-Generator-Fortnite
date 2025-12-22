import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset, Features, Value, Image
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import math


def training_function():
    # Configuración
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    dataset_name = "outfits_procesados_1024_rgb - copia/Food"
    resolution = 1024
    train_batch_size = 1
    num_train_epochs = 5
    learning_rate = 1e-4
    max_grad_norm = 1.0
    mixed_precision = "fp16"

    # Inicializar acelerador
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=mixed_precision,
    )

    # Cargar tokenizer y text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )

    # Cargar modelo VAE y UNet
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    # Congelar parámetros
    pipeline.vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Configurar LoRA
    attn_procs = {}
    for name in pipeline.unet.attn_processors.keys():
        attn_procs[name] = LoRAAttnProcessor2_0()

    pipeline.unet.set_attn_processor(attn_procs)

    # Optimizador
    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=learning_rate,
    )

    # Cargar dataset
    features = Features({
        "image": Image(),
        "text": Value("string"),
        "instance_prompt": Value("string"),
        "class_prompt": Value("string"),
    })
    dataset = load_dataset(
        "json",
        data_files=os.path.join(dataset_name, "metadata.jsonl"),
        features=features
    )

    # Preparar dataloader
    def collate_fn(examples):
        print(examples)
        input_ids = [example["text"] for example in examples]
        pixel_values = [example["image"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer(
            input_ids,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

    train_dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Preparar para el entrenamiento
    pipeline.unet, optimizer, train_dataloader = accelerator.prepare(
        pipeline.unet, optimizer, train_dataloader
    )

    # Entrenamiento
    progress_bar = tqdm(range(num_train_epochs * len(train_dataloader)))
    global_step = 0

    for epoch in range(num_train_epochs):
        pipeline.unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(pipeline.unet):
                # Forward pass
                latents = pipeline.vae.encode(
                    batch["pixel_values"]
                ).latent_dist.sample()
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                )
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predicción del ruido
                noise_pred = pipeline.unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Calcular pérdida
                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        pipeline.unet.parameters(), max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if global_step % 100 == 0:
                # Guardar checkpoint
                accelerator.wait_for_everyone()
                unwrapped_unet = accelerator.unwrap_model(pipeline.unet)
                unwrapped_unet.save_pretrained(
                    f"lora_checkpoint_{global_step}",
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )

    # Guardar modelo final
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(pipeline.unet)
    unwrapped_unet.save_pretrained(
        "lora_final",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )


if __name__ == "__main__":
    training_function()

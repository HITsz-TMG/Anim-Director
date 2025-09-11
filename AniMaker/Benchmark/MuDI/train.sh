VAL_PROMPT="olis harry potter toy and hta corgi playing together, floating on the pool."
DATASET_NAME="actionfigure_2_and_dog0"
VAE_PATH="/data/model/sdxl-vae-fp16-fix"
SDXL_PATH="/data/model/stable-diffusion-xl-base-1.0"
SEG_MIX_PROB=0.3
SEG_MIX_START_STEP=0

accelerate launch train_segmix_lora_sdxl.py --instance_data_dir="dataset/category/"$DATASET_NAME \
    --pretrained_model_name_or_path=$SDXL_PATH \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --output_dir=output/$DATASET_NAME"_p"$SEG_MIX_PROB"_s"$SEG_MIX_START_STEP \
    --validation_prompt="$VAL_PROMPT" --report_to=wandb --resolution=1024 --train_batch_size=1 --gradient_accumulation_steps=2 \
    --checkpointing_steps=20000 --max_train_steps=2010 --validation_epochs=30 --save_steps=200 --lr_warmup_steps=0 --seed=42 --rank=32 \
    --learning_rate=1e-4 \
    --segmix_prob=$SEG_MIX_PROB --segmix_start_step=$SEG_MIX_START_STEP --relative_scale=0.0 --soft_alpha 1.0 --gradient_checkpointing
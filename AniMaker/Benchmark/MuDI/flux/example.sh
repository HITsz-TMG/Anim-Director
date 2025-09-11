export MODEL_NAME="/data/model/FLUX.1-dev"
export TRAIN_CONFIG_PATH="flux/teddys"
export WANDB_NAME="test"

accelerate launch flux/train_segmix_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME --mixed_precision="bf16" \
  --instance_data_dir=$TRAIN_CONFIG_PATH --caption_column="text" \
  --resolution=512  --learning_rate=3e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 --rank=32 \
  --max_train_steps=2010 --checkpointing_steps=20000 --save_steps=500 --validation_epochs 40 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="output_flux/"$WANDB_NAME \
  --validation_prompt="a olis beige teddy bear and hta brown teddy bear dancing in the disco party, 4K, high quality" --report_to="wandb" --wandb_name=$WANDB_NAME --guidance_scale=1 \
  --segmix_prob=0.3 --segmix_start_step=0

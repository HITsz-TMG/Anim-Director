DATA_PATH=Benchmark/Inceptionnext/data/moeimouto-faces
CODE_PATH=Benchmark/Inceptionnext # modify code path here


# ALL_BATCH_SIZE=32
# NUM_GPU=1
# GRAD_ACCUM_STEPS=1 # Adjust according to your GPU numbers and memory size.
# let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS
NUM_GPU=1
GRAD_ACCUM_STEPS=1
BATCH_SIZE=32


MODEL=inceptionnext_tiny
DROP_PATH=0.1


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt adamw --lr 4e-3 --warmup-epochs 5 \
--batch-size $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DROP_PATH
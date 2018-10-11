# Shell script for training seq2seq Model
# Basic Variables for easy manupilation
export VOCAB_SOURCE=${HOME}/Thesis/AutoVis/Data/vocab/vocab.source
export VOCAB_TARGET=${HOME}/Thesis/AutoVis/Data/vocab/vocab.target
export TRAIN_SOURCES=${HOME}/Thesis/AutoVis/Data/train/train.sources
export TRAIN_TARGETS=${HOME}/Thesis/AutoVis/Data/train/train.targets
export DEV_SOURCES=${HOME}/Thesis/AutoVis/Data/dev/dev.sources
export DEV_TARGETS=${HOME}/Thesis/AutoVis/Data/dev/dev.targets

export DEV_TARGETS_REF=${HOME}/Thesis/AutoVis/Data/dev/dev.targets
export TRAIN_STEPS=100000

# RUNNING PYTHON SCRIPT
export MODEL_DIR=${HOME}/Thesis/AutoVis/Model
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./configuration/config.yml,
      ./configuration/train_seq2seq.yml,
      ./configuration/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_delimiter: ''
      target_delimiter: ''
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_delimiter: ''
       target_delimiter: ''
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR \
  tensorboard --logdir $MODEL_DIR
# END OF FILE

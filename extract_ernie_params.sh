#!/usr/bin/env bash

function print_help() {
  echo "Usage: $0 ERNIE_MODEL_NAME"
  echo "  ERNIE_MODEL_NAME the model identifier. Allowed identifiers are"
  echo "    ERNIE_Base_en_stable-2.0.0"
  echo "    ERNIE_Large_en_stable-2.0.0"
  exit 3
}

TF_MAJOR=1

while [ "$1" ]; do
  case "$1" in
  -h|--help)
  print_help
  ;;
  -*)
    print_help
  ;;
  *)
    ERNIE_MODEL_NAME="$1"
    break
  esac
done

case "${TF_MAJOR}" in
  1|2)
  ;;
  *)
    print_help
  ;;
esac


if [ -z "$ERNIE_MODEL_NAME" ]; then
  print_help
fi

DUMP_PATH=./model_artifacts/$ERNIE_MODEL_NAME
echo "Extracting parameters from $ERNIE_MODEL_NAME"

if [ -e $DUMP_PATH/${ERNIE_MODEL_NAME}_persistables.pkl ]; then
 echo "Old $DUMP_PATH/${ERNIE_MODEL_NAME}_persistables.pkl exists, please move it away first"
 exit 1
fi

pip3 install -r extractor-requirements.txt

python3 ernie4us/extract_ernie_params.py --ernie_path=./model_artifacts/$ERNIE_MODEL_NAME

echo "copying artifacts..."
mkdir -p $DUMP_PATH
cp ./model_artifacts/$ERNIE_MODEL_NAME/paddle/vocab.txt $DUMP_PATH/${ERNIE_MODEL_NAME}_vocab.txt
cp ./model_artifacts/$ERNIE_MODEL_NAME/paddle/ernie_config.json $DUMP_PATH/${ERNIE_MODEL_NAME}_config.json
mv ./model_artifacts/$ERNIE_MODEL_NAME/ernie_persistables.pkl $DUMP_PATH/${ERNIE_MODEL_NAME}_persistables.pkl

ls -lth $DUMP_PATH

if [ -e $DUMP_PATH/${ERNIE_MODEL_NAME}_persistables.pkl ]; then
 echo "$ERNIE_MODEL_NAME parameters are exported to $DUMP_PATH"
else
 echo "Failed to extract $ERNIE_MODEL_NAME parameters"
 exit 2
fi

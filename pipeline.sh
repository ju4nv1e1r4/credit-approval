#!/bin/bash

# Stop execution if any command fails
set -e

# Define colors
GREEN="\033[1;32m"
CYAN="\033[1;36m"
RESET="\033[0m"

echo -e "${CYAN}Criando novas features...${RESET}"
python3 src/feature_engineer/features.py

echo -e "${CYAN}Pré-pŕocessando os dados...${RESET}"
python3 src/models/preprocess.py

echo -e "${CYAN}Treinando o modelo...${RESET}"
python3 src/models/train.py

echo -e "${GREEN}Pipeline executado com sucesso!${RESET}"

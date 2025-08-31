# AGENTS Instructions

Este repositório visa revisar e otimizar a pipeline de treino do **SDXL**.

## Objetivo Principal
- Revisar a pipeline de treino do SDXL, com foco especial nos módulos **DoRA/LoRA**.
- Identificar pontos de otimização usando ferramentas de *profiling*.
- Desenvolver *addons* em linguagem **CUDA** de baixo nível para acelerar partes críticas.
- Carregar esses *addons* através do **PyTorch**.

## Orientações
- Não é necessário instalar dependências.
- Não devem ser executados testes automatizados.
- As alterações serão testadas localmente pelo usuário, pois dependem de GPU.


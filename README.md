# 🛰️ SpaceData Triage - Módulo GAIE

Bem-vindo ao repositório de Inteligência Artificial Generativa e Machine Learning (GAIE) do projeto **SpaceData Triage**. 

Como estudante do 4º ano de Engenharia de Software, desenvolvi este módulo para resolver um problema real da nova economia espacial: o gargalo de transmissão de dados. Com milhares de satélites em órbita, enviar dados corrompidos para a Terra desperdiça banda, tempo e dinheiro. A solução? Uma IA embarcada que filtra a telemetria e descarta o que é ruído antes mesmo da transmissão.

---

## 1. Definição do Problema e Qualidade dos Dados
O problema que nossa IA resolve é a **Triagem Preditiva de Telemetria**. O objetivo é classificar se um pacote de dados gerado por um satélite é "Útil" (válido para análise) ou "Ruído" (corrompido por anomalias espaciais).

Para treinar o modelo, criei um script Python que gerou um dataset sintético com **1.000 linhas e 9 features técnicas**, simulando sensores orbitais (temperatura, radiação, tensão da bateria, taxa de erros, latência, etc.). Os dados foram rotulados logicamente: pacotes expostos a alta radiação, baixa tensão ou alta taxa de erros foram marcados como ruído (`0`), enquanto o restante foi marcado como válido (`1`).

## 2. Pré-processamento e Engenharia de Atributos
Os dados foram estruturados utilizando a biblioteca `pandas`. Como os dados foram gerados sinteticamente com controle de distribuição (usando `numpy.random`), não houve valores nulos para tratar. 

A principal engenharia de atributos ocorreu na criação da variável alvo `Dado_Valido`, que consolidou as regras de negócio de hardware espacial em uma única métrica binária, permitindo que o modelo realizasse a classificação supervisionada de forma clara.

## 3. Aplicação, Comparação de Modelos e Validação
Para garantir a melhor performance, treinei e comparei dois modelos distintos separando os dados em 80% para treino e 20% para teste:

* **Regressão Logística:** Utilizado como modelo base (baseline). Apresentou uma acurácia de **76.50%**.
* **Random Forest Classifier:** Escolhido como modelo principal devido à sua excelente capacidade de lidar com múltiplas variáveis e regras de decisão não lineares.

🏆 **Resultado:** O Random Forest obteve uma acurácia de **99.50%** no conjunto de teste, provando ser altamente confiável para a missão.

## 4. Interpretabilidade com SHAP
Para não termos um modelo "caixa preta", utilizei a biblioteca **SHAP (SHapley Additive exPlanations)** para entender exatamente como a IA toma suas decisões de descarte.

> 📸 **[COLOQUE A IMAGEM DO SHAP AQUI]**
> ![Análise SHAP](output.png)

**Análise do Gráfico:**
Como podemos observar no *summary plot* acima, o modelo aprendeu perfeitamente a física do problema:
* **Tensão da bateria:** Valores baixos (bolinhas azuis) puxam a predição para a esquerda (descartar).
* **Taxa de erros e Radiação:** Valores altos (bolinhas vermelhas) puxam a predição fortemente para a esquerda, indicando que o modelo sabe que a radiação cósmica corrompe os sensores.
* Variáveis irrelevantes como desgaste e vibração normal ficaram agrupadas no centro, não interferindo no peso da decisão.

## 5. Deploy da Aplicação
Para demonstrar o modelo em funcionamento, desenvolvi uma interface interativa utilizando **Streamlit**. Através dela, é possível simular a leitura dos sensores do satélite e ver a IA executando a triagem em tempo real.

> ![Dados aprovados!](image.png)

> ![Dados descartados!](image-1.png)

---

## 💻 Instruções para Execução

Para rodar este projeto e testar a interface de triagem na sua máquina, siga os passos abaixo:

**1. Clone este repositório:**
```bash
git clone [INSERIR_O_LINK_DO_SEU_GITHUB_AQUI]
cd GS_IA

#Instale as dependencias necessarias
pip install -r requirements.txt

#Execute o app streamlit
#O aplicativo abrirá automaticamente no seu navegador padrão na porta 8501
python -m streamlit run app_streamlit/app.py

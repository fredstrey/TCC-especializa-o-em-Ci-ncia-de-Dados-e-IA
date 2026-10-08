# Previsão de Probabilidade de Acidentes com Aprendizado de Máquina

Trabalho de Conclusão de Curso da Especialização em Inteligência Artificial e Ciência de Dados (UFES / UnAC).

- **Aluno:** Frederico Luiz Strey
- **Orientador:** Prof. Ph.D. Alexandre Loureiros Rodrigues
- **Artigo completo:** [TCC Ciência de Dados.pdf](TCC%20Ci%C3%AAncia%20de%20Dados.pdf)

## Sobre o projeto

O TCC foi desenvolvido no formato de uma competição no Kaggle (**Competição TCC IA 2025**), com dados anonimizados de uma seguradora de veículos. A tarefa é treinar um modelo que estime o risco de acidente: para cada condutor em uma janela de tempo, prever a probabilidade de ocorrer um acidente na janela seguinte.

É um problema de classificação binária bastante desbalanceado (acidentes são cerca de 1,5% dos registros), avaliado pela **área sob a curva ROC (AUC)**, métrica oficial da competição.

## Dados

Cada linha representa um condutor em uma janela de tempo, com um `id`, o `target` binário (houve ou não acidente na janela seguinte) e 157 variáveis explicativas. Os dados já foram entregues pré-processados e anonimizados pela organização, então os nomes das variáveis não revelam diretamente o seu significado, apenas o grupo a que pertencem:

| Prefixo | Grupo |
|---|---|
| `cd` | Comportamento de direção (telemetria do veículo em movimento) |
| `temp` | Características temporais do período observado |
| `dem` | Características demográficas do condutor ou do contrato |
| `regiao` | Características das regiões de circulação do veículo |
| `hist` | Histórico de eventos anteriores do condutor ou do veículo |

O sufixo `prep` indica que a variável passou por algum pré-processamento (normalização, padronização ou codificação categórica).

A competição fornece três arquivos, que **não estão neste repositório**:

- `base_treinamento.csv`: features e `target`
- `base_teste.csv`: mesmas features, sem o `target`
- `exemplo_submissao.csv`: modelo do arquivo de submissão

## Metodologia

Foram comparados três algoritmos: **Regressão Logística**, **Random Forest** e **XGBoost**.

1. **Comparação inicial** dos três modelos com hiperparâmetros padrão, combinados com diferentes técnicas de balanceamento de classes: sem balanceamento, Random UnderSampling, Random OverSampling e SMOTE (biblioteca `imbalanced-learn`), além da ponderação de classes.
2. **Otimização de hiperparâmetros** com `RandomizedSearchCV`, validação cruzada estratificada em 3 folds e AUC ROC como critério.
3. **Segunda rodada de otimização** do XGBoost, com intervalos de busca ampliados e ajuste fino, executada em GPU.
4. **Treino final** dos três modelos com os melhores hiperparâmetros e geração da submissão.

Todos os modelos usam o mesmo pipeline:

- `SimpleImputer(strategy='median')` para valores faltantes
- `StandardScaler` para padronização
- PCA opcional (a busca escolheu `passthrough`, ou seja, sem redução de dimensionalidade, nos três modelos)

## Resultados

### Efeito do balanceamento

Sem balanceamento, os modelos atingem acurácia próxima de 0,98 com recall e F1 próximos de zero: preveem quase tudo como "não acidente". Na Regressão Logística, por exemplo:

| Balanceamento | AUC | Acurácia | Recall |
|---|---|---|---|
| Nenhum | 0,6679 | 0,9851 | 0,0003 |
| RandomOverSampler | 0,6673 | 0,6621 | 0,5836 |
| RandomUnderSampler | 0,6647 | 0,6479 | 0,5953 |

O balanceamento reduz a acurácia para níveis realistas e faz o modelo passar a identificar os casos de acidente, com AUC praticamente inalterada. Por isso a acurácia não serve como métrica neste problema, e a AUC foi usada em todas as comparações. A tabela completa, com os três modelos, está no artigo.

### Modelos otimizados

AUC ROC na validação (holdout estratificado de 20%):

| Modelo | AUC ROC | Melhores hiperparâmetros |
|---|---|---|
| Regressão Logística | 0,6931 | `C=0.6605`, `penalty='l2'`, `solver='lbfgs'` |
| XGBoost | 0,6924 | `n_estimators=369`, `max_depth=3`, `learning_rate=0.038`, `subsample=0.6746`, `colsample_bytree=0.8298`, `gamma=0`, `min_child_weight=1`, `reg_alpha=1`, `reg_lambda=5`, `scale_pos_weight=1` |
| Random Forest | 0,6420 | `n_estimators=487`, `max_depth=11`, `min_samples_leaf=3`, `min_samples_split=3` |

![Curva ROC dos modelos](auc.png)

Regressão Logística e XGBoost ficaram praticamente empatados na validação, e o Random Forest ficou bem abaixo dos dois. O XGBoost foi o modelo escolhido para a submissão oficial, alcançando **0,68752** de AUC no leaderboard da competição.

A configuração vencedora do XGBoost combina árvores rasas (`max_depth=3`), taxa de aprendizado baixa e regularização L1/L2, o que favorece a generalização em uma base com 157 features anonimizadas, possivelmente redundantes e correlacionadas.

## Conteúdo do repositório

| Arquivo | Descrição |
|---|---|
| [notebook7b238902e8.ipynb](notebook7b238902e8.ipynb) | Notebook de treinamento otimizado: treina os três modelos com os melhores hiperparâmetros, mede a AUC na validação e gera o `submission.csv` |
| [TCC Ciência de Dados.pdf](TCC%20Ci%C3%AAncia%20de%20Dados.pdf) | Artigo do TCC |
| [auc.png](auc.png) | Curva ROC dos três modelos |

## Como executar

O notebook foi escrito para rodar no Kaggle, lendo os dados de `/kaggle/input/competicao-tcc-ia-2025/`.

1. Abra o notebook no Kaggle e adicione os dados da competição como input.
2. Ative a GPU: o XGBoost está configurado com `device='cuda'`.
3. Execute todas as células.

Para rodar localmente, ajuste os caminhos dos CSVs e, se não houver GPU, troque `device='cuda'` por `device='cpu'`. Dependências:

```bash
pip install numpy pandas scikit-learn xgboost joblib
```

A execução salva os três modelos treinados (`modelo_randomforest.pkl`, `modelo_logistic.pkl`, `modelo_xgboost.pkl`) e o arquivo `submission.csv`, com a probabilidade de acidente para cada `id` da base de teste.

> **Nota:** a versão do notebook neste repositório calcula o `scale_pos_weight` do XGBoost pela proporção entre as classes, o que resulta em AUC de 0,6864 na validação. O valor de 0,6924 reportado no artigo e no gráfico corresponde a `scale_pos_weight=1`. Além disso, o `submission.csv` gerado por esta versão é a média das probabilidades da Regressão Logística e do XGBoost.

## Limitações e trabalhos futuros

- Os recursos computacionais limitaram a extensão da busca de hiperparâmetros e a variedade de modelos testados.
- Como as variáveis são anonimizadas, não foi possível fazer engenharia de atributos orientada pelo domínio.
- Próximos passos possíveis: buscas de hiperparâmetros mais amplas, outros algoritmos de boosting e estratégias de ensemble.

## Referências

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.
- Chen, T.; Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16, 785–794.
- Hosmer, D. W.; Lemeshow, S.; Sturdivant, R. X. (2013). *Applied Logistic Regression*. Wiley, 3ª ed.
- Lemaitre, G.; Nogueira, F.; Aridas, C. K. (2017). *imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning*.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

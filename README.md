# Data-Science: Predicción de Falla por Fatiga en Superaleaciones

### Planteamiento del Problema

La predicción del crecimiento de grietas por fatiga en superaleaciones base níquel resulta de interés
para ámbitos de ingeniería y gestión del riesgo. Una correcta aplicación de modelación supervisada
permitiría optimizar el diseño de componentes, planificar el mantenimiento predictivo y prevenir fallos
inesperados. Esto es especialmente importante en la industria aeroespacial y de energía, donde la
falla de un componente puede tener consecuencias desastrosas tanto en términos de seguridad
como de costos.

### Objetivo

Desarrollar un modelo de predicción supervisado para estimar si el tamaño de las grietas bajo fatiga
en superaleaciones base níquel se encuentra en el Modo de Falla III. El modelo debe lograr una
precisión mínima del 90% en la clasificación de estas grietas durante las pruebas de validación,
utilizando datos del Departamento de Ciencias de Materiales y Metalurgia de la Universidad de
Cambridge y aplicaremos técnicas avanzadas de machine learning para desarrollar, entrenar y
validar el modelo. El objetivo es optimizar los flujos de inspección y retiro de componentes críticos,
reduciendo el riesgo de fallas catastróficas y mejorando tanto la seguridad como la eficiencia
operativa.

### Descripción de los Datos

El conjunto de datos fuente proviene del Grupo de Transformaciones de Fase del Departamento de
Ciencias de Materiales y Metalurgia de la Universidad de Cambridge y se encuentra constituido por
1894 registros sobre la tasa de crecimiento de grietas por fatiga en superaleaciones a base de níquel
en función de 51 variables.

La tabla Analítica de datos está constituida por las variables anteriormente mencionadas.
Adicionalmente se generaron las siguientes variables:

- cambio da/dN : Representa la razón de cambio en el crecimiento de la grieta.
Dicha variable es de particular interés dado que se relaciona estrechamente con la ecuación de
Paris, que son un conjunto de relaciones empíricas que describen la tasa de propagación de grietas
bajo cargas cíclicas en materiales frágiles y dúctiles.

Mismas que indican que una grieta se encuentra en Modo Falla II cuando el crecimiento de la grieta
es estable con una relación casi lineal en escala logarítmica entre da/dN y ΔK. Gracias a la
determinación de la variable anterior se genero un criterio de para un clasificador binario que permita
determinar si la grieta se encuentra en un mecanismo de Modo de Falla III. Dicha variable fungirá
como target dada la importancia de lograr predecir si una grieta se encuentra un proceso inestable
de crecimiento y esto sería un indicador de próxima fractura.

El criterio para determinar si la grieta está en Modo Falla III se determinó usando el histograma de
la variable cambio_da/dN donde se observa que la mayoría de los datos se encuentran muy cercanos
a 0 tomándose como indicador de crecimiento estable (Modo Falla II) por lo anterior si la grieta
presenta un cambio mayor a 490 μm se cataloga como Modo Falla III.

### Limpieza de Datos

El set de datos analizado demuestra no tener presencia de valores nulos; por lo cual no se realizan
procesos de imputación. El tratamiento de outliers no se aplicará debido a que la presencia de estos
puede tener relevancia en el fenómeno a estudiar por lo cual en el preprocesamiento se realiza un
escalamiento de datos.
Las siguientes tablas muestran las principales variables correlacionadas con la variable objetivo
Modo Falla III (tanto positivas como negativas):

Como se observa ninguna presenta una correlación mayor a 0.90 por lo cual no se eliminan variables,
sin embargo, al generar el mapa de calor para entender la correlación entre las variables se detectan
casos con niveles alto de correlación. Se decide eliminar las variables 'Minimum grain size',
'Maximum grain size' porque tienen una alta correlación entre ellas sin embargo no presentan una
correlación importante con la variable objetivo.

Adicional en la TAD se excluyen las variables 'da/dN', ‘?K’ y se conservan las variables que
representan a su logaritmo, dado que en estudios de mecánica de fractura ambos datos son más
significativos en su forma logarítmica y adicionalmente permite tener una representación de
distribución normal de dichas variables.

Otra consideración en la TAD fue eliminar la variable ‘cambio_da/dN’ pues la target proviene de una
transformación de la misma.

A continuación se muestran las variables utilizadas para la modelación así como el preprocesamiento
aplicado en un pipeline:

### EDA Análisis Exploratorio de Datos (Principales Insights)

A continuación, se describen los principales insights relacionados al negocio a partir del
EDA.
- Los datos muestran un comportamiento que sigue un fenómeno físico esperado, conforme
a los estudios de mecánica de la fractura. Según estos estudios, a mayores niveles de
tensión (expresados en log ∆K), se observa un incremento en el crecimiento de las grietas
(expresado en log da/dN). Este hallazgo es clave para comprender y predecir el
comportamiento de las grietas bajo diferentes condiciones de carga.

- La frecuencia (vibraciones) muestra tener una efecto inverso en la relación log da/dN vs log?K

- Se detectan puntos se interés que presenta altos niveles de intensidad de tensión y un tamaño de
grieta elevado, si bien esos puntos se encuentran con un mecanismo de falla tipo III es relevante ver
que aún en dichas condiciones tienen un límite de fluencia mayor a 1500 MPa, un espesor de
muestra menor a 5 mm y los elementos de aleación que muestran estar relacionados con estás
condiciones extremas de comportamiento es la presencia de Y2O3 (Óxido de Itrio), Ta, W, Zr, B,
altos niveles de Cr y Ni.

### Modelación Supervisada

Se realiza modelación supervisada para realizar una clasificación binaría que permita predecir si
una grieta se encuentra en Modo de Falla III (1) o no (0).

### Métricas de Modelación

La identificación de los casos positivos dentro de este modelo tiene una mayor ponderación dando
mayor importancia a la sensibilidad (recall) que a la precisión. La métrica recall permite medir la
proporción de verdaderos positivos entre todos los casos positivos reales, y es importante cuando el
costo de los falsos negativos es alto.
Igualmente se evaluará el AUC-ROC, métrica que mide la capacidad del modelo para distinguir entre
clases. Esto con el objetivo de obtener una visión más completa del rendimiento del modelo.

### Selección del Mejor Modelo

La finalidad de la modelación supervisada es lograr predecir cuando una grieta tiene un
comportamiento en modo de falla tipo III dado que en esté punto se encuentra próxima una fractura,
lo cual podría resultar en una catástrofe por lo cual es imperativo que se clasifique correctamente la
clase 1, el principal factor para determinar el mejor modelo fue encontrar aquel que muestre el mejor
desempeño de predicción en la clase de interés esto teniendo en cuenta que la métrica recall mide
la proporción de verdaderos positivos entre los casos positivos reales. Adicionalmente se evaluó
AUC-ROC en el set de prueba y se complemento con una validación cruzada.

El modelo LGBMClassifier mostro una buena capacidad para identificar ambas clases con un recall
en la clase 1 de 0.92 y en clase 0 de 0.94, fue el modelo que logró predecir mejor la clase 1 sin
perder poder de predicción de la clase 0. El AUC-ROC en el set de prueba y en la validación de
cruzada fueron consistentes demostrando que el modelo es capaz de discriminar entre las clases
correctamente teniendo solo una variación de 0.84% entre ambas. Es posible observar que el modelo
no presenta signos de sobreajuste durante el entrenamiento.

Parámetros LGBMClassifier utilizados:

'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_samples': 40,
'n_estimators': 100, 'num_leaves': 20, 'scale_pos_weight': 1, 'subsample': 0.8

### Reporte de estabilidad de los modelos

PSI (Population Stability Index)

El PSI (Population Stability Index) obtenido fue de 0.10 indicando un cambio relativamente pequeño
en la distribución de las puntuaciones. Aunque los bins de las puntuaciones tienen un
comportamiento simular, presentan un desplazamiento hacia la izquierda sugiriendo que el conjunto
OTT se está sesgando hacia 1, esto puede estar siendo provocado porque el modelo está optimizado
para maximizar el recall.

CSI (Characteristic Stability Index)

Durante la prueba se detectan variables que están generando un cambio en las distribuciones. En el primer
caso se evalúan las distribuciones previo al preprocesamiento de los datos, y la gráfica posterior es posterior a
la transformación detectando que se logro reducir el cambio en la distribución de las variables sin embargo es
importante notar que se sigue detectando un cambio de distribución entre set y set por lo cual puede conllevar
a reducir el rendimiento del modelo, o indicar un sesgo en la selección de muestra.

Kolmogorov-Smirnov (KS)

Los resultados de la prueba KS Statistic: 0.8250008105566904, p-value: 3.38299598311626e-77
indicando que existen una diferencia entre las distribuciones de las dos categorías evaluadas en los
conjuntos de datos y por el valor de p-value se entiende que esta distribución no obedece aleatoridad
por lo cual el modelo muestra una buena capacidad para distinguir entre la clase 1 y 0.

True Error (TE) y True Negative Error (TNE)

TE (True Error): Es la proporción de falsos negativos (FN) sobre la suma de falsos negativos y
verdaderos positivos (TP). Aproximadamente el 10.74% de los casos positivos fueron
incorrectamente clasificados como negativos

True Negative Error (TNE): Es la proporción de falsos positivos (FP) sobre la suma de falsos
positivos y verdaderos negativos (TN). Aproximadamente el 7.25% de los casos negativos fueron
incorrectamente clasificados como positivos.

Interpretabilidad (Shap Values)

De acuerdo a los resultados podemos observar las variables que tienen una mayor contribución en
la predicción del modelo.

### Caso de uso del modelo

Para determinar el punto de corte óptimo en la curva ROC se aplicó el método Youden para equilibrar
la sensibilidad y la especificidad de un modelo de clasificación binaria. Determinando un punto de
corte óptimo en 0.41, esto con la finalidad de generar una estrategia donde se equilibre la detección
temprana de una grieta con un comportamiento Modo de Falla III y de esta forma evitar
consecuencias de no detectar grietas que podrían estar cerca de la fractura (falsos negativos), así
prevenir como los costos asociados con intervenciones innecesarias o reparaciones anticipadas
(falsos positivos).

Se generaron 3 clases de estudio para implementar acciones preventivas por el área encargada de
inspección y mantenimiento del negocio:

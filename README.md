# Контроль за вниманием локомотивной бригады
Решение
Решение кейса представляет собой прототип системы, способной анализировать видеофайлы на предмет отвлечения локомотивной бригады от управления на посторонние предметы (основное использование телефона или любого другого гаджета) и формировать по результатам анализа отчёт.
Отчёт должен содержать: количество выявленных нарушений, типы нарушений, время нарушения на временной шкале видеофайла.
При этом система должна определять именно факт использования телефона, а не наличие его в кадре. Не является нарушение кратковременное использование телефона, к примеру, активация экрана до 3х секунд для ознакомления со временем. 
## ХАРАКТЕРИСТИКИ
- Работает с изображением, видео
- Возможность подключения камеры
- Отличает 80 объектов
- Пользователь может указать, какой объект искать на изображении
 загрузать видео в этом коде video_path = "/Users/kakotichi/Downloads/train_dataset_Бригады/Анализ бригад (телефон)/Есть телефон/02_59_18.mp4"



# Data Science SF
Привет Мир! Я Алина и я изучаю Data Science in SkillFactory.


# Проект: Контроль за вниманием локомотивной бригады. 
# Содержание
1. [File description](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#File-description)
2. [Job description](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#Job-description)
3. [Quality metric](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#Quality-metric)
4. [Stages of the project](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#Stages-of-the-project)
5. [Result](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#Result)

## Описание файлов
 (файл и что содержит)
- *1_Data_collection_and_preparation.ipynb* - first stage of work: collection, processing and preliminary analysis of the submitted data, creation of a summary table
- *2_FE_EDA_FS.ipynb* - the second stage of the work is the creation of new features, EDA, selection of features for clustering
- *3_Clustering.ipynb* - the third stage of work: clustering of regions, description of clusters, answers to the questions posed
- *project_module.py* - separate module with necessary functions and standardized region names
- *conda_env.yml* - if you use Anaconda (Conda)
- *requirements.txt* - if you use 'pip'
#
- *./project_chatbot* - solution realization in production (chat bot)       
- *./data_provided* - presented data to solve the clustering task      
- *./data_output* - output data obtained during the execution of the task:       
     - *data_final* - final summary table with features       
     - *data_final_clust* - final summary table with cluster labels       
     - *features_for_clustering.json* - features selected for clustering        
     - *.svg* and *.html* - polar diagram files for presentation in GitHub     

## Job description

We have data on income, morbidity, socially vulnerable segments of the Russian population and other economic and demographic data at our disposal.

**Our task:**
- cluster Russia's regions and determine which of them are most in need of assistance to low-income/disadvantaged segments of the population;
- describe population groups facing poverty;
- determine:
    - whether the number of children, pensioners and other socially vulnerable groups affects the level of poverty in the region;
    - whether the level of poverty/social disadvantage is related to production and consumption in the region;
    - what other dependencies can be observed in relation to socially disadvantaged segments of the population.

## Quality metric

To perform clustering of Russian regions, which will be effective in identifying regions that are in dire need of assistance to low-income/disadvantaged groups of the population. Answer the questions posed.

## Stages of the project

- Collection and processing of submitted data
- Downloading and processing of additional data
- Data processing and preliminary analysis
- feature creation (Feature Engineering)
- EDA
- Feature Selection
- Clustering
- Description of the clusters and answers to the questions posed

## Result

- Clustering performed
- Answers to the questions posed
- The solution is realized in production (chek the folder *./project_chatbot*)
- Project files uploaded to GitHub.


Описание файла

1_Data_collection_and_preparation.ipynb - первый этап работы: сбор, обработка и предварительный анализ представленных данных, создание сводной таблицы
2_FE_EDA_FS.ipynb - второй этап работы - создание новых функций, EDA, выбор функций для кластеризации
3_Clustering.ipynb - третий этап работы: кластеризация регионов, описание кластеров, ответы на заданные вопросы
project_module.py - отдельный модуль с необходимыми функциями и стандартизированными именами областей
conda_env.yml - если вы используете Anaconda (Conda)
requirements.txt - если вы используете 'pip'
./project_chatbot - реализация решения в производстве (чат-бот)
./data_provided - представлены данные для решения задачи кластеризации
./data_output - выходные данные, полученные во время выполнения задачи:
data_final - окончательная сводная таблица с функциями
data_final_clust - окончательная сводная таблица с метками кластеров
features_for_clustering.json - функции, выбранные для кластеризации
.svg и .html - файлы полярных диаграмм для презентации на GitHub
Описание вакансии

В нашем распоряжении есть данные о доходах, заболеваемости, социально уязвимых слоях населения России и другие экономические и демографические данные.

Наша задача:

кластеризация регионов России и определение того, какие из них больше всего нуждаются в помощи малообеспеченным/неблагополучным слоям населения;
описать группы населения, сталкивающиеся с бедностью;
определить:
влияет ли количество детей, пенсионеров и других социально уязвимых групп на уровень бедности в регионе;
связан ли уровень бедности/социального неблагоприятного положения с производством и потреблением в регионе;
какие еще зависимости можно наблюдать в отношении социально неблагополучных слоев населения.
Метрика качества

Для проведения кластеризации российских регионов, которая будет эффективной в определении регионов, которые остро нуждаются в помощи малообеспеченным/неблагополучным группам населения. Ответьте на поставленные вопросы.

Этапы проекта

Сбор и обработка представленных данных
Загрузка и обработка дополнительных данных
Обработка данных и предварительный анализ
создание функций (Инженерия функций)
ЭДА
Выбор функции
Кластеризация
Описание кластеров и ответы на заданные вопросы
Результат

Кластеризация выполнена
Ответы на заданные вопросы
Решение реализуется в производстве (переход папки ./project_chatbot)
Файлы проекта загружены на GitHub.

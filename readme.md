### Добро пожаловать на кейс РЖД - "Контроль за вниманием локомотивной бригады"!
***
Датасет упакован в файл ```dataset.zip``` и представляет из себя две папки: ```Есть телефон``` (72 файла) и ```Нет телефона``` (76 файлов).
В каждой из этих папок находятся видеофайлы, с ситуациями нерегламентированного использования мобильного телефона одним или несколькими членами локомотивной бригады. Нерегламентированным считается использование телефона в течение более чем 3 секунд.
Также, в датасете представлен пример результата обработки видеофайлов - файл ```submission.csv```. Он предназначен для проверки вашего решения на тестовом датасете, который будет предоставлен позже.
```submission.csv``` имеет разделитель ```,``` и следующую структуру:
```filename,cases_count,timestamps```, где:
1) ```filename``` - имя файла, по которому производится анализ
2) ```cases_count``` - количество ситуаций в одном файле (например, в видеофайле хотя бы один из членов локомотивной бригады один раз использует телефон дольше трех секунд, значит ```cases_count``` будет равным ```1```). Если одновременно телефон используется более чем одним человеком  - считать это одной ситуацией.
3) ```timestamps``` - временные метки ситуаций, если такие имеются (время считается относительно начала видеофайла, указывается в следующем формате - ```["MM:SS"]``` - при наличии одной ситуации и ```["MM:SS", "MM:SS"...]``` если ситуаций в файле больше). Внимание, временные метки должны быть упорядочены по возрастанию!
***
##### Вашей задачей является создание системы контроля бдительности локомотивной бригады, которая будет производить фиксацию отвлечения от управления на мобильный телефон. 

# ЖЕЛАЕМ УДАЧИ!




P.S. Не забудьте посетить экспертные сессии и не стесняйтесь задавать вопросы)
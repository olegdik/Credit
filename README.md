# Credit
Модель кредитного риск-менеджмента. 

### Обучение модели
Файл, обучающий модель - /model/pipeline2.py

Данные для обечения должны лежать в /model/data/

Обученное ядро модели записывается в /model/

### Записк сервиса
uvicorn.exe main:app --reload

main.py - основной сервиса, в котором реализовано следующие методы: 

predict - предсказание: 0 - если клиент вернет кредит вовремя и 1 - в противном случае.
Это POST запрос. Данные для предсказания передаются в формате json. Примеры данных лежат в /model/data/1.json,  /model/data/2.json

status - возвращает "I'm OK" если сервис запущен

version - информация о текущей версии сервиса

#### Файлы данных и обученная модель в формате .pkl не вошли в репозиторий, т.к. превышают отведенные в GitHub лимиты

# Репозиторий для отчетов и программ по предмету "Методы и алгоритмы принятия решений"

1. Делаем форк репо
2. Выполняем работу
3. Делаем пулл-реквест в мастер

Подробнее о git: https://denis-creative.com/poshagovaya-instruktsiya-po-rabote-s-git-i-github-dlya-studentov/

Тексты лабораторных: https://drive.google.com/drive/folders/122vocoWZNhzVsN6wnkw_j67W3h6PSm4C?usp=sharing

## Списки групп:

Отчеты коммитаем в **reports/<Ваша_фамилия>/<Номер_лабораторной_работы>/rep**

Код коммитаем в **reports/<Ваша_фамилия>/<Номер_лабораторной_работы>/src**

## Установка зависимостей

Реализовано для версии Python 3.9.5.

Установка под Linux:

```shell
python3.9 -m venv venv
source ./venv/bin/activate
pip3 install -r ./requirements.txt
```

Установка под Windows: разберетёсь))0)

Исходники лежат в **reports/Krupenkov/universal/src**,
рекомендуется пометить в PyCharm  чтобы не пыло проблем с импортами (*Mark Directory as > Sources Root*)

Ну и далее запускать скрипты вида `labXu.py`, где `X` - номер лабы.

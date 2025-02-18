# _Универсальная нейронная сеть_

## Принцип работы

#### В нейронную сеть записывается множество слоев с указанием для каждого

+ количества нейронов
+ F и F' активации

#### У класса NeuralNetwork прописаны методы:

```python
def learn(x: NDArray[NDArray], e: NDArray[NDArray], alpha: Optional[float])
```

_Сюда подается множество наборов входных параметров, эталонов и необязательная скорость обучения_

```python
def go(x: NDArray)
```

_Прогон входных параметров x_

```python
def prediction_results_table(x: NDArray[NDArray], e: NDArray[NDArray])
```

_Прогон множества массивов входных параметров и эталонов с красивым выводом_

## Возможности

Можно добавлять сколько угодно слоев, главное чтобы границы были одинаковы

```python
l1 = Layer(lens=(10, 4))
l2 = Layer(lens=(4, 5))
l3 = Layer(lens=(5, 2))
nn = NeuralNetwork(l1, l2, l3)
```

## Ограничения

1. Алгоритм медленнее "чистых" аналогов с константным количеством слоев, нейронов и прописанными вручную формулами, однако это улучшается наследуемыми от класса Layer слоями с определенными функциями активации (формулами, в где они раскрыты)
2. Изменять и улучшать достаточно сложно
3. У полной функциональности есть обратная сторона - избыточность
4. Ого, вы тут читаете! Ну с днем декораторов
5. Их тут нет...
6. Я уже говорил тебе, что такое безумие? Безумие - это точное повторение одного и того же действия, раз за разом, в надежде на изменение. Это   есть   безумие. Смысл в том - окей? И тогда я вижу это везде, везде, куда ни глянь. Я тут сижу над этой пятой лабой. Запускаю десятки тысяч образов. Раз за разом. Что-то меняется? НЕТ! На что я тогда надеюсь? Что эта нейронка как Иисус воскреснет и будет со 100-процентной точностью определять, какой номер у этого образа? Все в порядке... Я успокоюсь, братец, успокоюсь. Смысл в том... Я уже говорил тебе, что такое безумие?

# nsga2_sample

## Pythonについて
- `random.uniform(low, high)`: low <= x < highのxを1つ生成
- `tools.Statistics(lambda ind: ind.fitness.values)`
  - lambda: 無名関数
  - lambda x1, x2, ...: 式

## Deapについて
### toolbox使い方
`register(alias, method[, argument[, ...]])`
```
def func(a, b, c=3):
    print(a, b, c)

tools = Toolbox()
tools.register("myFunc", func, 2, c=4)
tools.myFunc(3)
>>> 2 3 4
```
- registerではdefault定数を設定できる
  - 今回は`a=2`と`c=4`

## 参考文献
1. [Deap: Github](https://github.com/DEAP/deap)
2. [Deap: Pythonと機械学習](http://darden.hatenablog.com/entry/2017/04/18/225459)
3. [Pythonの進化計算ライブラリDeap: Qiita](https://qiita.com/neka-nat@github/items/0cb8955bd85027d58c8e)
4. [Pythonの進化計算ライブラリDeap(2): Qiita](https://qiita.com/neka-nat@github/items/bf95041366b9f2c6171b)
5. [Deapのメモ: nobUnagaの日記](http://nobunaga.hatenablog.jp/entry/2016/05/16/121409)
6. [多目的最適化: Pythonと機械学習](http://darden.hatenablog.com/entry/2017/05/26/234845)
7. [statistics document: deap](http://deap.readthedocs.io/en/master/tutorials/basic/part3.html)
8. [進化方法一覧: deap](http://deap.readthedocs.io/en/master/api/tools.html)

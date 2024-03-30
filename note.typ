
3.

```
python BigramTester.py -f small_model.txt -t data/kafka.txt
Read 24944 words. Estimated entropy: 13.46
```

```
python BigramTester.py -f kafka_model.txt -t data/small.txt
Read 19 words. Estimated entropy: 10.29
```

(c)

```
❯ python .\BigramTester.py -f .\guardian_model.txt -t .\data\guardian_test.txt
Read 871878 words. Estimated entropy: 6.62
```

```
❯ python .\BigramTester.py -f .\austen_model.txt -t .\data\austen_test.txt
Read 10738 words. Estimated entropy: 6.97
```
# Made 2021. LSML. Введение в Spark.

В файлах `.zpln` и `.ipynb` представлен один и тот же семинар.

Для поднятия Zeppelin локально можно использовать официальный [Docker](https://zeppelin.apache.org/download.html#using-the-official-docker-image).
Пример:

```bash
docker run -d -p 7077:7077 -p 8080:8080 -p 4040:4040 -v $PWD/notebook:/notebook -e ZEPPELIN_NOTEBOOK_DIR="/notebook" apache/zeppelin:0.10.0
```

Проект `sbt` можно запустить через `sbt run` или собрать в jar через `sbt package`, после чего можно запустить на спарке через `spark-submit`

```bash
spark-submit --class ru.made.SparkWordCount --master spark://localhost:7077 target/scala-2.12/spark-examples_2.12-0.1.jar
```
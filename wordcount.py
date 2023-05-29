su root
cd /mnt/spark/bin

nano new.txt

nano wordcount.scala

import org.apache.spark.SparkContext

var sc = new SparkContext()
val map = sc.textFole("/mnt/spark/bin/new.txt").flatMap(line =>line.split(" ")).map(word => (word,1));

val counts = map.reduceByKey(_ + _)
counts.count
counts.collect

val keysRdd = counts.keys
val sorted = keysRdd.sortBy(x => x,true)
sorted.collect

counts.saveAsTextFile("/mnt/spark/bin/hey1")
sorted.saveAsTextFile("/mnt/spark/bin/hey2")

spark-shell

:load wordcount.scala
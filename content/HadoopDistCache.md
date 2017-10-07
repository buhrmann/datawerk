Title: Reading from distributed cache in Hadoop
Category: Data Posts
Tags: hadoop, big data, java
Slug: hadoop-distributed-cache
Authors: Thomas Buhrmann

The distributed cache can be used to make small files (or jars etc.) available to mapreduce functions locally on each node. This can be useful e.g. when a global stopword list is needed by all mappers for index creation.  Here are two correct ways of reading a file from distributed cache in Hadoop 2. This has changed in the new API and very few books and tutorials have updated examples.

###Named File###

In the driver:

```java
Job job = Job.getInstance(new Configuration());
job.addCacheFile(new URI ("/path/to/file.csv" + "#filelabel"));
```

In the mapper:
```java
@Override
public void setup(Context context) throws IOException, InterruptedException
{
  URI[] cacheFiles = context.getCacheFiles();
  if (cacheFiles != null && cacheFiles.length > 0)
  {
    try
    {
      BufferedReader reader = new BufferedReader(new FileReader("filelabel"));
    }
...
}
```

###File system###

In the driver:
```java
Job job = Job.getInstance(new Configuration());
job.addCacheFile(new URI ("/path/to/file.csv"));
...
```

In the mapper:
```java
@Override
public void setup(Context context) throws IOException, InterruptedException
{
  URI[] cacheFiles = context.getCacheFiles();
  if (cacheFiles != null && cacheFiles.length > 0)
  {
    try
    {
        FileSystem fs = FileSystem.get(context.getConfiguration());
        Path path = new Path(cacheFiles[0].toString());
        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));
    }
...
}
```
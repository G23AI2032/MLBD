import java.io.IOException;
import java.util.StringTokenizer;
import java.util.logging.*;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCountQ9 {
  private static final Logger LOGGER = Logger.getLogger(WordCount.class.getName());

  public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      String line = value.toString();
      StringTokenizer itr = new StringTokenizer(line);
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken());
        context.write(word, one);
      }
    }
  }

  public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: WordCount <input_path> <output_path>");
      System.exit(2);
    }

    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd_HHmmss");
    String timestamp = dateFormat.format(new Date());
    String logFileName = "wordcount_performance_" + timestamp + ".log";

    FileHandler fileHandler = new FileHandler(logFileName);
    fileHandler.setFormatter(new SimpleFormatter());
    LOGGER.addHandler(fileHandler);
    LOGGER.setLevel(Level.INFO);

    LOGGER.info("Starting WordCount performance testing");
    LOGGER.info("Input path: " + args[0]);
    LOGGER.info("Base output path: " + args[1]);

    long[] splitSizes = {1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024};

    for (long splitSize : splitSizes) {
      Configuration conf = new Configuration();
      conf.setLong("mapreduce.input.fileinputformat.split.maxsize", splitSize);

      Job job = Job.getInstance(conf, "word count - split size: " + splitSize);
      job.setJarByClass(WordCount.class);
      job.setMapperClass(TokenizerMapper.class);
      job.setCombinerClass(IntSumReducer.class);
      job.setReducerClass(IntSumReducer.class);
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(IntWritable.class);

      String outputPath = args[1] + "_" + (splitSize / (1024 * 1024)) + "MB";

      FileInputFormat.addInputPath(job, new Path(args[0]));
      FileOutputFormat.setOutputPath(job, new Path(outputPath));

      LOGGER.info("\nStarting job with split size: " + splitSize + " bytes");

      long startTime = System.currentTimeMillis();

      boolean success = job.waitForCompletion(true);

      long endTime = System.currentTimeMillis();
      long executionTime = endTime - startTime;
      long numMaps = job.getCounters().findCounter("org.apache.hadoop.mapred.Task$Counter", "MAP_INPUT_RECORDS").getValue();

      StringBuilder results = new StringBuilder();
      results.append("\nJob Statistics for split size: ").append(splitSize).append(" bytes\n");
      results.append("Split Size: ").append(splitSize).append(" bytes (").append(splitSize / (1024 * 1024)).append("MB)\n");
      results.append("Execution Time: ").append(executionTime).append(" ms\n");
      results.append("Number of Maps: ").append(numMaps).append("\n");
      results.append("Average time per map: ").append(executionTime / numMaps).append(" ms\n");
      results.append("----------------------------------------");

      LOGGER.info(results.toString());

      if (!success) {
        LOGGER.severe("Job failed for split size: " + splitSize);
        System.exit(1);
      }
    }

    LOGGER.info("WordCount performance testing completed");
    fileHandler.close();
  }
}
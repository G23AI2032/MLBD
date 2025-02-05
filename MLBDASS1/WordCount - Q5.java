import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

  public static class TokenizerMapper
       extends Mapper<Object, Text, Text, IntWritable>{

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      // Log the input received by the map function
      System.out.println("Map Input: " + value.toString());

      // Remove punctuation from the input
      String sanitizedLine = value.toString().replaceAll("\\p{P}", "");
      System.out.println("Sanitized Input: " + sanitizedLine);
      // Tokenize the sanitized input
      StringTokenizer itr = new StringTokenizer(sanitizedLine);
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken());
        context.write(word, one);
        // Log each output emitted by the map function
        System.out.println("Map Output: " + word.toString() + ", 1");
      }
    }
  }

  public static class IntSumReducer
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      // Log the reduce input for this key
      System.out.print("Reduce Input for key " + key.toString() + ": ");
      int sum = 0;
      for (IntWritable val : values) {
        // Log each individual value in the reduce phase
        System.out.print(val.get() + " ");
        sum += val.get();
      }
      System.out.println();
      result.set(sum);
      context.write(key, result);
      // Log the output from the reduce phase
      System.out.println("Reduce Output: " + key.toString() + ", " + sum);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "word count");
    job.setJarByClass(WordCount.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
package wordcount;

import java.io.IOException;
import java.util.StringTokenizer;
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

public class WordCount {

    public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
        private Text word = new Text();
        private IntWritable count = new IntWritable();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] parts = line.split("\\s+");
            if (parts.length >= 2) {
                word.set(parts[0]);  // First part is the word
                count.set(Integer.parseInt(parts[1]));  // Second part is the count
                context.write(word, count);
            }
        }
    }

    public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            for (IntWritable val : values) {
                context.write(key, val);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount.class);
        
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);
        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        // Add timing measurement
        long startTime = System.currentTimeMillis();
        
        // Configure split size (experiment with different values)
        conf = job.getConfiguration();
        conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 671088640); // 64MB default, adjust this value
        
        // Run the job and capture success
        boolean success = job.waitForCompletion(true);
        
        // Calculate and display execution time
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        System.out.println("Total execution time: " + executionTime + " milliseconds");
        
        System.exit(success ? 0 : 1);
    }
}
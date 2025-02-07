import java.util.*;

public class Question3 {
    public static void main(String[] args) {
        
        Map<String, List<Integer>> inputPairs = new HashMap<>();
        inputPairs.put("up", Arrays.asList(1, 1, 1, 1));
        inputPairs.put("to", Arrays.asList(1, 1, 1));
        inputPairs.put("get", Arrays.asList(1, 1));
        inputPairs.put("lucky", Arrays.asList(1));

        
        System.out.println("Input Pairs for Reduce Phase:");
        for (Map.Entry<String, List<Integer>> entry : inputPairs.entrySet()) {
            System.out.println("(" + entry.getKey() + ", " + entry.getValue() + ")");
        }
        System.out.println();
        
        
        Map<String, Integer> outputPairs = new HashMap<>();
        
        for (Map.Entry<String, List<Integer>> entry : inputPairs.entrySet()) {
            String key = entry.getKey();
            List<Integer> values = entry.getValue();
            
            int sum = 0;
            for (int value : values) {
                sum += value;
            }
            
            outputPairs.put(key, sum);
            System.out.println("Key: " + key);
            System.out.println("Input values: " + values);
            System.out.println("Reduce Output: " + key + " -> " + sum);
            System.out.println();
        }
    }
}
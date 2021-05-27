import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.Triple;
import javax.json.*;
import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonObject;
import javax.json.JsonObjectBuilder;
import java.io.*;
import java.sql.*;
import java.util.*;
import java.util.stream.Collectors;

public class ExtractNER {
    private AbstractSequenceClassifier<CoreLabel> model;

    private List<String> testCorpus;


    public ExtractNER(String modelPath) throws IOException, ClassNotFoundException {
        model = CRFClassifier.getClassifier(modelPath);
    }

    public static JsonArrayBuilder namedEntityList2Json(List<NamedEntity> entities) {
        JsonArrayBuilder rsult = Json.createArrayBuilder();
        for (NamedEntity e: entities) {
            rsult.add(e.toJsonObject().build());
        }
        return rsult;
    }

    public List<NamedEntity> predict(String sentence) {
        List<NamedEntity> result = new ArrayList<>();
        model.classifyToCharacterOffsets(sentence).forEach(triple -> {
            String entityType = triple.first();
            int offset0 = triple.second();
            int offset1 = triple.third();
            String name = sentence.substring(offset0, offset1);
            result.add(new NamedEntity(name, entityType, offset0, offset1));
        });
        return result;
    }


    public void setModel(AbstractSequenceClassifier<CoreLabel> model) {
        this.model = model;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        Properties config = new Properties();
        config.load(new BufferedReader(new FileReader("config.properties")));
//        String path = config.getProperty("tweets_path");
        String news_path = config.getProperty("news_path");
        String modelPath = config.getProperty("model_path");
        String resultPath = config.getProperty("result_path");
//        JsonArray tweetsJsonArray = EntityLinker.readJsonData(path);
        JsonArray newsJsonArray = EntityLinker.readJsonData(news_path);
        ExtractNER ner = new ExtractNER(modelPath);
        // for tweets
//        List<Pair<Integer, List<NamedEntity>>> tweetsCollect = tweetsJsonArray.parallelStream().map(val -> {
//            JsonObject obj = (JsonObject) val;
//            int index = obj.getInt("id");
//            String tweets = obj.getString("ner_tweets");
//            List<NamedEntity> entities = ner.predict(tweets);
//            System.out.println(tweets + ":" + Arrays.toString(entities.toArray()));
//            return new Pair<>(index, entities);
//        }).collect(Collectors.toList());

//        try (BufferedWriter writer = new BufferedWriter(new FileWriter(resultPath + "ne_tweets_clean.txt"))) {
//            for (Pair<Integer, List<NamedEntity>> pair : tweetsCollect) {
//                JsonObjectBuilder builder = Json.createObjectBuilder();
//                builder.add("id", pair.first);
//                builder.add("tweet",namedEntityList2Json(pair.second));
//                writer.write(builder.build().toString() + "\n");
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        // for news
        List<Pair<Integer, List<NamedEntity>>> newsCollect = newsJsonArray.parallelStream().map(val -> {
            JsonObject obj = (JsonObject) val;
            int index = obj.getInt("id");
            String news = obj.getString("news");
            List<NamedEntity> entities = ner.predict(news);
            return new Pair<>(index, entities);
        }).collect(Collectors.toList());

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(resultPath + "ne_news.txt"))) {
            for (Pair<Integer, List<NamedEntity>> pair : newsCollect) {
                JsonObjectBuilder builder = Json.createObjectBuilder();
                builder.add("id", pair.first);
                builder.add("news",namedEntityList2Json(pair.second));
                writer.write(builder.build().toString() + "\n");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }



//        List<NamedEntity> temp = ner.predict("realdonaldtrump everyone please keep calling your senators this week and continue to pray for the kavanagh family");
//        System.out.println(Arrays.toString(temp.toArray()));
    }

}


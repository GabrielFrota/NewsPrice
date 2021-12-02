package core;

import java.net.ProxySelector;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse.BodyHandlers;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.TreeMap;

import com.google.gson.JsonParser;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;

class NewsPrice {
  
  private class SentimentValue {
    final String date;
    final Double value;
    
    SentimentValue(String d, Double v) {
      date = d;
      value = v;
    }
  }
  
  final StanfordCoreNLP coreNLP;
  final TreeMap<String, Double> variations = new TreeMap<>();
  final ArrayList<SentimentValue> sentiments = new ArrayList<>();
  
  NewsPrice() {
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize, ssplit, pos, parse, sentiment");
    coreNLP = new StanfordCoreNLP(props);
  }
  
  int processNLP(String text) {
    var anno = coreNLP.process(text);
    var map = anno.get(CoreAnnotations.SentencesAnnotation.class);
    int score = 0;
    if (map.size() == 1) {
      var sent = map.get(0);
      var tree = sent.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
      score = RNNCoreAnnotations.getPredictedClass(tree) - 2;
    } else {
      for (var sent : map) {
        var tree = sent.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
        var curr = RNNCoreAnnotations.getPredictedClass(tree) - 2;
        score = Math.abs(curr) > Math.abs(score) ? curr : score;
      }
    }
    return score;
  }
  
  final String urlP = "https://query1.finance.yahoo.com/v7/finance/download/TSLA?"
      + "period1=1630022400&period2=1638316800&interval=1d&events=history&includeAdjustedClose=true";
  
  void loadPrices() throws Exception {
    var path = Paths.get("TSLA.csv");
    System.out.println("Querying Tesla's stock prices from last 3 months...");
    var req = HttpRequest.newBuilder().uri(new URI(urlP))
        .GET().timeout(Duration.ofSeconds(10)).build();
    var res = HttpClient.newBuilder().proxy(ProxySelector.getDefault())
        .build().send(req, BodyHandlers.ofFile(path));
    if (res.statusCode() != 200)
      throw new Exception("Something went wrong, response code is " + res.statusCode());
    
    var prices = Files.lines(path).skip(1).map(l -> {
      var vals = l.split(",");
      var date = vals[0];
      var price = vals[vals.length - 2];
      var out = date + "\t" + price;
      System.out.println(out);
      return out;
    }).toArray(String[]::new);
    
    for (int i = 1; i < prices.length; i++) {
      var prev = prices[i - 1].split("\t");
      var curr = prices[i].split("\t");
      var vtion = Double.parseDouble(curr[1]) - Double.parseDouble(prev[1]);
      variations.put(curr[0], vtion);
    }
    System.out.println("Price variation values");
    for (var v : variations.entrySet()) {
      System.out.println(v.getKey() + "\t" + v.getValue());
    }
  }
  
  final String[] d0 = {"20210901", "20210930"};
  final String[] d1 = {"20211001", "20211031"};
  final String[] d2 = {"20211101", "20211126"};
  final String[][] dates = {d0, d1, d2};
  final String key = "3hwf3bUnRWNpKjaIkNy0iPxAWIKjp3kJ";
  final String urlN = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q=Tesla&api-key=" + key; 
  
  void loadNews() throws Exception {  
    System.out.println("Sentiment scale (-2 = very negative, -1 = negative, 0 = neutral, 1 = positive, 2 = very positive)");
    System.out.println("Querying most relevant Tesla news from last 3 months and running CoreNLP "
        + "sentiment analysis tools in their abstract...");
    
    for (int i = 0; i < 3; i++) {
      for (int page = 0; page < 2; page++) {
        var query = urlN + "&begin_date=" + dates[i][0] + "&end_date=" + dates[i][1] 
            + "sort=relevance" + "&page=" + Integer.toString(page);
        var req = HttpRequest.newBuilder().uri(new URI(query))
            .GET().timeout(Duration.ofSeconds(10)).build();
        var res = HttpClient.newBuilder().proxy(ProxySelector.getDefault())
            .build().send(req, BodyHandlers.ofString());
        var body = JsonParser.parseString(res.body());
        var docs = body.getAsJsonObject().get("response").getAsJsonObject().get("docs");
        for (var d : docs.getAsJsonArray()) {
          var abst = d.getAsJsonObject().get("abstract").getAsString();
          var date = d.getAsJsonObject().get("pub_date").getAsString().split("T")[0];
          System.out.println(date + "\t" + abst);
          var val = processNLP(abst);
          System.out.println("\t\tSentiment level: " + val);
          sentiments.add(new SentimentValue(date, (double) val));  
        }
      }
    }
  }
  
  private void printArrs(double[] arr1, double[] arr2) {
    for (int i = 0; i < arr1.length; i++) {
      System.out.println(arr1[i] + "\t" + arr2[i]);
    }
  }
  
  void calcCorrelation() {
    var sments = new double[sentiments.size()];
    var vtions = new double[sentiments.size()];
    var entries = new ArrayList<Entry<String, Double>>();
    int cnt = 0;
    for (var s : sentiments) {
      sments[cnt] = s.value;
      var e = variations.ceilingEntry(s.date);
      entries.add(e);
      vtions[cnt] = e.getValue();
      cnt++;
    }
    System.out.println("==============================");
    System.out.println("Values for stock price in the same day of news date");
    printArrs(sments, vtions);
    System.out.println("pearson correlation value is " + ArrayMath.pearsonCorrelation(sments, vtions));
    
    cnt = 0;
    for (var e : entries) {    
      var prev = variations.lowerEntry(e.getKey());
      vtions[cnt] = variations.lowerEntry(prev.getKey()).getValue();
      cnt++;
    }
    System.out.println("==============================");
    System.out.println("Values for stock price 2 days before news date");
    printArrs(sments, vtions);
    System.out.println("pearson correlation value is " + ArrayMath.pearsonCorrelation(sments, vtions));
    
    cnt = 0;
    for (var e : entries) {
      vtions[cnt] = variations.lowerEntry(e.getKey()).getValue();
      cnt++;
    }
    System.out.println("==============================");
    System.out.println("Values for stock price 1 day before news date");
    printArrs(sments, vtions);
    System.out.println("pearson correlation value is " + ArrayMath.pearsonCorrelation(sments, vtions));
    
    cnt = 0;
    for (var e : entries) {
      vtions[cnt] = variations.higherEntry(e.getKey()).getValue();
      cnt++;
    }
    System.out.println("==============================");
    System.out.println("Values for stock price 1 day after news date");
    printArrs(sments, vtions);
    System.out.println("pearson correlation value is " + ArrayMath.pearsonCorrelation(sments, vtions));
    
    cnt = 0;
    for (var e : entries) {
      var next = variations.higherEntry(e.getKey());
      vtions[cnt] = variations.higherEntry(next.getKey()).getValue();
      cnt++;
    }
    System.out.println("==============================");
    System.out.println("Values for stock price 2 days after news date");
    printArrs(sments, vtions);
    System.out.println("pearson correlation value is " + ArrayMath.pearsonCorrelation(sments, vtions));
  }
  
  public static void main (String[] args) throws Exception {
    var n = new NewsPrice();
    n.loadPrices();
    n.loadNews();
    n.calcCorrelation();
  }  
  
}

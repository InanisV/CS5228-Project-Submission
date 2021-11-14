## Description Processor based on Word2Vec

How to use:

1. `pip install gensim`

2. Download Google's Word2Vec pre-trained model to `/data`

   ```shell
   wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
   
   gzip -d GoogleNews-vectors-negative300.bin.gz
   ```

3. Modify locations of data files as needed 
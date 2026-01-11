# Online Üretici Yapay Zeka, Doğal Dil İşleme ve Büyük Dil Modelleri (350 saat)


Açıklama: Bu kursumuz üretici ağlar, doğal dil işleme ve büyük dil modellerini çeşitli düzeylerde kapsamlı bir biçimde ele almaktadır. Kursumuz yapay zeka alanında devrimsel bir etkiye yol açan bu teknolojilerin teorik altyapıları ve gerçekleştirimleri konusunda uygulamalı bilgiler vermeyi ve beceriler kazandırmayı hedeflemektedir. 

* __Gözden Geçirme ve Hazırlık__
   * Yapay Zeka, Makine Öğrenmesi ve Veri Bilimi İçin Neden Python? 
   * Yapay Zeka, Makine Öğrenmesi ve Veri Biliminde Diğer Programlama Dillerinin Avantajları ve Dezavantajları
   * Python Programlama Dilinin Yeniden Gözden Geçirilmesi
   * Scipy, Numpy, Pandas ve Matplotlib Kütüphanelerinin Genel Tanıtımı
   * SciPy ve scikit-learn Kütüphanelerinin Genel Tanıtımı
   * Tensorflow, Keras ve PyTorch Kütüphanelerinin Genel Tanıtımı
   * HuggingFace Kütüphanelerinin Genel Tanıtımı

* __Giriş ve Temel Bilgiler__
  * Makine Öğrenmesi Nedir?
  * Makine Öğrenmesinin Çeşitleri: Denetimli (Supervised) Öğrenme, Denetimsiz (Unsupervised) Öğrenme ve Pekiştirmeli (Reinforcement) Öğrenme Kavramları
  * Makine Öğrenmesinin Uygulama Alanları
  * Makine Öğrenmesinin Diğer Disiplinlerle İlgisi
  * Makine Öğrenmesi İçin Ne Kadar Matematik ve İstatistik Bilmek Gerekir?
  * Veri Bilimi ile Yapay Zeka ve Makine Öğrenmesi Arasındaki İlişki
  * Veri Madenciliği Nedir?
  * Doğal Dil İşleme Medir?

* __Metinlerin Atomlarına Ayrılması (Tokenizastion)__
  * Sözcük Dağarcığı (Vocabulary), Derlem (Corpus) ve Doküman (Document) Kavramları
  * Atom Nedir? Atomlara Ayırma Ne Anlama Gelmektedir?
  * Karakter Düzeyinde Atomlarına Ayırma (Character Level Tokenization)
  * Sözcük Düzeyinde Atomlarına Ayırma (Word Level Tokenization)
  * Altsözcük Düzeyinde Atomlarına Ayırma (Subword Tokenization)
  * Boşluklardan ve Noktalama İşaretlerinden Faydalanarak Atomlarına Ayırma Yönteminin Gerçekeleştirimi
  * Kural Tabanlı Atomlarına Ayırma İşlemleri Gerçekelştirimi
  * Altsözcük Düzeyinde Atomlarına Ayırma (Subword Tokenization) Yöntemlerinin Gerçekleştirimleri
  * Byte Çiftlerinin Kodlanması (Byte Pair Encoding) Yönteminde Atomlara Ayırma
  * WordPiece Yöntemi 
  * NLTK Kütüphanesi İle Atomlarına Ayırma
  * SpaCy Kütüphanesi İle Atomlarına Ayırma İşlemleri
  * Hagging Face Kütüphanesi İle Atomlara Ayırma İşlemleri
  * SentencePiece Kütüphanesi İle Atomlara Ayırma İşlemleri
  * Zemberek Kütüphanesi İle Atomlara Ayırma İşlemleri

* __Atomların Sayısal Biçimde Temsil Edilmesi__
  * Atomlar Neden Sayısal Biçimde Temsil Edilmektedir?
  * One-hot-encoding ve Sözcük Çantası (Bag of Words) Yöntemleri
  * Sözcük Çantası (Bag of Words) Yönteminin Manuel Gerçekleştirimi
  * Sözcük Çantası (Bag of Words) Yönteminin scikit-learn Kütüphanesiyle Gerçekleştirilmesi
  * TF-IDF (Term Frequency-Inverse Document Frequency) Yöntemi
  * TF-IDF Yönteminin manuel ve scikit-learn Kütühanesi Kullanılarak Gerçekleştirilmesi
  * N-Gram Modeller (Unigram, Bigram, Trigram) ve Bağlamsal Bilginin Yakalanması
  * N-Gram Modellerin Manuel ve scikit-learn Kütüphanesi Kullanılarak Gerçekleştirilmesi
  * Byte Çiftlerinin Kodlanması Yönteminin Manuel Gerçekleştirimi    
  * WordPiece Yönteminin Hugging Face Kütüphanesi Kullanılarak Gerçekleştirilmesi
  * SentencePiece Kütüphanesi İle Sayısal Temsillerin Oluşturulması
  * Hugging Face Kütüphanesi İle Saysal Temsillerin Oluşturulması
  * Sözcük Gömme (Word Embedding) Kavramı
  * Word2Vec Algoritması
  * GloVe Algoritması
  * FastText Algoritması
  * Sözcük Gömme İşlemlerinin Kütüphaneler Yoluyla Gerçekleştirilmesi
  * Bağlamsal Gömme İşlemleri (Contextualized Embeddings)

# NLP Kurs İçeriği - Tokenization ve Encoding

## BÖLÜM 1: Metinlerin Token'lara Ayrılması (Tokenization)

### 1.1 Temel Kavramlar
- Vocabulary (Sözcük Dağarcığı)
- Corpus (Derlem)
- Document (Doküman)
- Token vs Type Kavramları

### 1.2 Token Düzeyleri
- Karakter Düzeyinde Tokenization
- Sözcük Düzeyinde Tokenization
- Altsözcük Düzeyinde Tokenization (Subword)

### 1.3 Temel Tokenization Yöntemleri
- Boşluk ve Noktalama Tabanlı Ayırma
- Kural Tabanlı Tokenization
- Manuel Gerçekleştirim

### 1.4 Subword Tokenization Algoritmaları

#### 1.4.1 Byte Pair Encoding (BPE)
- Algoritma Mantığı
- Merge Operations
- Manuel Gerçekleştirim

#### 1.4.2 WordPiece
- BPE'den Farkları
- Likelihood-based Merging
- Manuel Gerçekleştirim

#### 1.4.3 Unigram Language Model
- Top-down Yaklaşım
- Probabilistic Pruning

### 1.5 Kütüphaneler ile Tokenization
- NLTK Kütüphanesi İle Tokenization
- SpaCy Kütüphanesi İle Tokenization
- Hugging Face Tokenizers Kütüphanesi İle Tokenization
- SentencePiece Kütüphanesi İle Tokenization (BPE ve Unigram Modları)
- Zemberek Kütüphanesi İle Tokenization (Türkçe için)

---

## BÖLÜM 2: Token'ların Sayısal Biçimde Temsil Edilmesi (Encoding)

### 2.1 Neden Sayısal Temsil?
- Makine Öğrenmesi Modelleri ve Vektör Uzayı
- Benzerlik Hesaplama (Cosine Similarity, Euclidean Distance)

### 2.2 Klasik Yöntemler

#### 2.2.1 One-Hot Encoding ve Bag of Words (BoW)
- One-Hot Encoding Mantığı
- Bag of Words Yaklaşımı
- Avantajları ve Dezavantajları
- Manuel Gerçekleştirim
- scikit-learn Kütüphanesi ile Gerçekleştirim

#### 2.2.2 TF-IDF (Term Frequency-Inverse Document Frequency)
- TF (Term Frequency) Hesaplama
- IDF (Inverse Document Frequency) Hesaplama
- Normalization Teknikleri
- Manuel Gerçekleştirim
- scikit-learn Kütüphanesi ile Gerçekleştirim

#### 2.2.3 N-Gram Modeller
- Unigram, Bigram, Trigram Kavramları
- Bağlamsal Bilginin Yakalanması
- Karakter vs Kelime N-gramları
- Manuel Gerçekleştirim
- scikit-learn Kütüphanesi ile Gerçekleştirim

### 2.3 Subword Token Encoding

#### 2.3.1 BPE Token'larının Encoding'i
- Vocabulary → ID Mapping
- Token'dan Sayıya Dönüşüm
- Manuel Gerçekleştirim
- Hugging Face Tokenizers ile Kullanım

#### 2.3.2 WordPiece Token'larının Encoding'i
- BERT Tokenizer Kullanımı (Hugging Face Transformers)
- Special Tokens: [CLS], [SEP], [PAD], [MASK], [UNK]
- Attention Masks
- Input Preparation

#### 2.3.3 SentencePiece ile Encoding
- BPE Mode
- Unigram Mode
- Model Eğitimi ve Kullanımı
- Vocabulary Yönetimi

#### 2.3.4 Tiktoken (OpenAI GPT Tokenization)
- GPT-2/3/4 Tokenizer Kullanımı
- Token Sayısı Hesaplama
- API Token Limitleri

### 2.4 Word Embeddings (Sözcük Gömme)

#### 2.4.1 Temel Kavramlar
- Sparse vs Dense Representations
- Distributional Semantics
- Embedding Space Özellikleri
- Dimensionality Reduction vs Learned Embeddings

#### 2.4.2 Word2Vec Algoritması
- Skip-gram Modeli
- CBOW (Continuous Bag of Words)
- Basitleştirilmiş Manuel İmplementasyon
- Gensim Kütüphanesi ile Gerçekleştirim
- Pre-trained Modeller (Türkçe Embeddings)
- Similarity ve Analogy Testleri

#### 2.4.3 GloVe (Global Vectors) Algoritması
- Co-occurrence Matrix
- Word2Vec'e Göre Farkları
- Pre-trained GloVe Modelleri Kullanımı

#### 2.4.4 FastText Algoritması
- Subword Information Kullanımı
- Character N-grams
- OOV (Out-of-Vocabulary) Handling
- Gensim Kütüphanesi ile Gerçekleştirim

#### 2.4.5 Pratik Uygulamalar
- Türkçe Corpus Üzerinde Embedding Eğitimi
- Kelime Analogileri (kral - erkek + kadın = ?)
- Word Similarity Testleri
- Embedding Visualization (t-SNE, PCA)

### 2.5 Document ve Sentence Embeddings

#### 2.5.1 Basit Aggregation Yöntemleri
- Average Pooling
- Weighted Average (TF-IDF Weighting)
- Max/Min Pooling
- Manuel İmplementasyon

#### 2.5.2 Doc2Vec (Paragraph Vect

# Kursa Kayıt
[Kursumuza ön kayıt yaptırmak için bu bağlantıyı kullanabilirsiniz.](https://us02web.zoom.us/meeting/register/A2N5yVVXStuiYN0tAXTeJw#/registration)


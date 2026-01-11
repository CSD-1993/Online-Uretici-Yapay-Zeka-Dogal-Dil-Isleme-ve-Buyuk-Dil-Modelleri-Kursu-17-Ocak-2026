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

* __Metinlerin Atomlarına Ayrılması (Tokenization)__
  * Temel Kavramlar
    - Sözcük Dağarcığı (Vocabulary)
    - Derlem (Corpus)
    - Doküman (Document)
    - Atom vs Tip Kavramları (Token vs Type)
      
  * Atom Düzeyleri
    - Karakter Düzeyinde Atomlarına Ayırma
    - Sözcük Düzeyinde Atomlarına Ayırma
  - Altsözcük Düzeyinde Atomlarına Ayırma (Subword)

  * Temel Atomlarına Ayırma Yöntemleri
    - Boşluk ve Noktalama Tabanlı Atomlarına Ayırma
    - Kural Tabanlı Atomlarına Ayırma
    - Manuel Gerçekleştirim

### Altsözcük Düzeyinde Atomlarına Ayırma Algoritmaları

#### Byte Çiftlerinin Kodlanması (Byte Pair Encoding - BPE)
- Algoritma
- Birleştirme İşlemleri (Merge Operations)
- Manuel Gerçekleştirim

#### WordPiece Yöntemi
- BPE'den Farkları
- Olabilirlik Tabanlı Birleştirme (Likelihood-based Merging)
- Manuel Gerçekleştirim

#### Unigram Dil Modeli (Unigram Language Model)
- Yukarıdan Aşağıya Yaklaşım (Top-down Approach)
- Olasılıksal Budama (Probabilistic Pruning)

### Kütüphaneler ile Atomlarına Ayırma
- NLTK Kütüphanesi İle Atomlarına Ayırma
- SpaCy Kütüphanesi İle Atomlarına Ayırma
- Hugging Face Tokenizers Kütüphanesi İle Atomlarına Ayırma
- SentencePiece Kütüphanesi İle Atomlarına Ayırma (BPE ve Unigram Modları)
- Zemberek Kütüphanesi İle Atomlarına Ayırma (Türkçe için)

## Atomların Sayısal Biçimde Temsil Edilmesi

### Atomlar Neden Sayısal Biçimde Temsil Edilmektedir?
- Makine Öğrenmesi Modelleri ve Vektör Uzayı
- Benzerlik Hesaplama (Kosinüs Benzerliği, Öklid Uzaklığı)

### Klasik Yöntemler

#### One-Hot Encoding ve Sözcük Çantası (Bag of Words)
- One-hot-encoding Kodlaması
- Sözcük Çantası Yaklaşımı
- Avantajları ve Dezavantajları
- Manuel Gerçekleştirim
- scikit-learn Kütüphanesi ile Gerçekleştirim

#### TF-IDF (Terim Frekansı-Ters Doküman Frekansı)
- TF (Terim Frekansı) Hesaplama
- IDF (Ters Doküman Frekansı) Hesaplama
- Normalleştirme Teknikleri
- Manuel Gerçekleştirim
- scikit-learn Kütüphanesi ile Gerçekleştirim

#### N-Gram Modeller
- Unigram, Bigram, Trigram Kavramları
- Bağlamsal Bilginin Yakalanması
- Karakter vs Kelime N-gramları
- Manuel Gerçekleştirim
- scikit-learn Kütüphanesi ile Gerçekleştirim

### Altsözcük Atomlarının Sayısal Temsili

#### BPE Atomlarının Kodlanması
- Sözcük Dağarcığı → Kimlik Eşleşmesi (Vocabulary → ID Mapping)
- Atomdan Sayıya Dönüşüm
- Manuel Gerçekleştirim
- Hugging Face Tokenizers ile Kullanım

#### WordPiece Atomlarının Kodlanması
- BERT Tokenizer Kullanımı (Hugging Face Transformers)
- Özel Simgeler (Special Tokens): [CLS], [SEP], [PAD], [MASK], [UNK]
- Dikkat Maskeleri (Attention Masks)
- Girdi Hazırlama (Input Preparation)

#### SentencePiece ile Kodlama
- BPE Modu
- Unigram Modu
- Model Eğitimi ve Kullanımı
- Sözcük Dağarcığı Yönetimi

#### Tiktoken (OpenAI GPT Atomlarına Ayırma)
- GPT-2/3/4 Tokenizer Kullanımı
- Atom Sayısı Hesaplama
- API Atom Limitleri

### Sözcük Gömme (Word Embeddings)

#### Temel Kavramlar
- Seyrek vs Yoğun Temsiller (Sparse vs Dense Representations)
- Dağılımsal Anlambilim (Distributional Semantics)
- Gömme Uzayı Özellikleri
- Boyut İndirgeme vs Öğrenilmiş Gömmeler

#### Word2Vec Algoritması
- Skip-gram Modeli
- CBOW (Sürekli Sözcük Çantası - Continuous Bag of Words)
- Basitleştirilmiş Manuel İmplementasyon
- Gensim Kütüphanesi ile Gerçekleştirim
- Önceden Eğitilmiş Modeller (Pre-trained) (Türkçe Gömmeler)
- Benzerlik ve Analoji Testleri

#### GloVe (Küresel Vektörler) Algoritması
- Birlikte Görülme Matrisi (Co-occurrence Matrix)
- Word2Vec'e Göre Farkları
- Önceden Eğitilmiş GloVe Modelleri Kullanımı

#### FastText Algoritması
- Altsözcük Bilgisi Kullanımı (Subword Information)
- Karakter N-gramları
- Sözcük Dağarcığı Dışı Sözcük İşleme (OOV Handling)
- Gensim Kütüphanesi ile Gerçekleştirim

### Doküman ve Cümle Gömmeleri

#### Basit Birleştirme Yöntemleri (Aggregation)
- Ortalama Havuzlama (Average Pooling)
- Ağırlıklı Ortalama (TF-IDF Ağırlıklandırma)
- Maksimum/Minimum Havuzlama (Max/Min Pooling)
- Manuel Gerçekleştirimler

#### Doc2Vec (Paragraf Vektörleri)
- PV-DM ve PV-DBOW
- Gensim Kütüphanesi ile Kullanım

#### Cümle Dönüştürücüleri (Sentence Transformers - SBERT)
- SBERT Yaklaşımı
- sentence-transformers Kütüphanesi
- Önceden Eğitilmiş Modeller
- Anlamsal Arama Uygulamaları

### Bağlamsal Gömme İşlemleri (Contextualized Embeddings)

#### Statik ve Bağlamsal Gömmeler
- Çok Anlamlılık Problemi (Polysemy)
- Bağlama Bağlı Temsiller

#### 2BERT Gömmeleri (WordPiece Kullanımı)
- BERT Atomlarına Ayırma Hattı (Pipeline)
- Gömme Çıkarımı (Embedding Extraction)
- Hugging Face Transformers ile Kullanım
- BERTurk (Türkçe BERT)

#### GPT Gömmeleri (BPE Kullanımı)
- GPT Atomlarına Ayırma Hattı
- Byte Düzeyinde BPE
- Gömme Çıkarımı

####  Diğer Modern Modeller
- RoBERTa (BPE Kullanımı)
- T5 (SentencePiece Unigram)
- ALBERT (SentencePiece Unigram)

#### 2.7.1 Yöntemlerin Karşılaştırılması
- Sözcük Çantası/TF-IDF: Ne Zaman Kullanılmalı?
- Gömmeler: Ne Zaman Kullanılmalı?
- Altsözcük Atomlarına Ayırma: Optimal Sözcük Dağarcığı Boyutu

#### Değerlendirme Ölçütleri
- İçsel Değerlendirme (Intrinsic): Kelime Benzerliği, Analoji
- Dışsal Değerlendirme (Extrinsic): Alt Görev Performansı

## Kurs Çıktıları
Kurs sonunda katılımcılar:
1. Tüm önemli atomlarına ayırma ve kodlama yöntemlerini anlayacak
2. Sıfırdan gerçekleştirim yapabilecek
3. Ürün ortamına hazır (production-ready) kütüphaneleri kullanabilecek
4. Hangi yöntemin hangi problem için uygun olduğuna karar verebilecek
5. Türkçe için özel optimizasyonlar yapabilecek
6. Modern dönüştürücü modellerinin (transformer) atomlarına ayırma stratejilerini anlayacak
   
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


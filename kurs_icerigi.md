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

  * Byte Çiftlerinin Kodlanması (Byte Pair Encoding - BPE) Yöntemi
    - Algoritma
    - Birleştirme İşlemleri (Merge Operations)
    - Manuel Gerçekleştirim

  * WordPiece Yöntemi
    - BPE'den Farkları
    - Olabilirlik Tabanlı Birleştirme (Likelihood-based Merging)
    - Manuel Gerçekleştirim

  * Kütüphaneleri Kullanarak Atomlarına Ayırma
    - NLTK Kütüphanesi İle Atomlarına Ayırma
    - SpaCy Kütüphanesi İle Atomlarına Ayırma
    - Hugging Face Tokenizers Kütüphanesi İle Atomlarına Ayırma
    - SentencePiece Kütüphanesi İle Atomlarına Ayırma (BPE ve Unigram Modları)
    - Zemberek Kütüphanesi İle Atomlarına Ayırma (Türkçe için)

* __Atomların Sayısal Biçimde Temsil Edilmesi__
    * Temeller
      - Atomlar Neden Sayısal Biçimde Temsil Edilmektedir?
      - Makine Öğrenmesi Modelleri ve Vektör Uzayı
      - Benzerlik Hesaplama (Kosinüs Benzerliği, Öklid Uzaklığı)
    * One-hot-encoding ve Sözcük Çantası (Bag of Words) Yaklaşımı
      - Manuel Gerçekleştirim
      - scikit-learn Kütüphanesi ile Gerçekleştirim
    * TF-IDF (Terim Frekansı-Ters Doküman Frekansı)
      - TF (Terim Frekansı) Hesaplama
      - IDF (Ters Doküman Frekansı) Hesaplama
      - Normalleştirme Teknikleri
      - Manuel Gerçekleştirim
      - scikit-learn Kütüphanesi ile Gerçekleştirim
  * N-Gram Modeller
    - Unigram, Bigram, Trigram Kavramları
    - Bağlamsal Bilginin Yakalanması
    - Karakter vs Kelime N-gramları
    - Manuel Gerçekleştirim
    - scikit-learn Kütüphanesi ile Gerçekleştirim
  * BPE Atomlarının Kodlanması
    - Sözcük Dağarcığı → Kimlik Eşleşmesi (Vocabulary → ID Mapping)
    - Atomdan Sayıya Dönüşüm
    - Manuel Gerçekleştirim
    - Hugging Face Tokenizers ile Kullanım
  * WordPiece Yöntemi İle Uygulama
  * SentencePiece Kütüphanesi ile Uygulama
  * Tiktoken İle GPT Tarzı 
    - GPT-2/3/4 Tokenizer Kullanımı
    - Atom Sayısı Hesaplama
    - API Atom Limitleri
  * Sözcük Gömme (Word Embeddings) Yöntemleri
    - Seyrek vs Yoğun Temsiller (Sparse vs Dense Representations)
    - Dağılımsal Anlambilim (Distributional Semantics)
    - Gömme Uzayı Özellikleri
    - Boyut İndirgeme vs Öğrenilmiş Gömmeler
  * Word2Vec Algoritması
    - Skip-gram Modeli
    - CBOW (Sürekli Sözcük Çantası - Continuous Bag of Words)
    - Basitleştirilmiş Manuel İmplementasyon
    - Gensim Kütüphanesi ile Gerçekleştirim
    - Önceden Eğitilmiş Modeller (Pre-trained) (Türkçe Gömmeler)
    - Benzerlik ve Analoji Testleri
  * GloVe Algoritması
    - Birlikte Görülme Matrisi (Co-occurrence Matrix)
    - Word2Vec'e Göre Farkları
    - Önceden Eğitilmiş GloVe Modelleri Kullanımı
  * FastText Algoritması
    - Altsözcük Bilgisi Kullanımı (Subword Information)
    - Karakter N-gramları
    - Sözcük Dağarcığı Dışı Sözcük İşleme (OOV Handling)
    - Gensim Kütüphanesi ile Gerçekleştirim
  * Doküman ve Cümle Gömmeleri
    - Basit Birleştirme Yöntemleri (Aggregation)
    - Ortalama Havuzlama (Average Pooling)
    - Ağırlıklı Ortalama (TF-IDF Ağırlıklandırma)
    - Maksimum/Minimum Havuzlama (Max/Min Pooling)
    - Manuel Gerçekleştirimler
* Diğer Modern Modeller
  - RoBERTa (BPE Kullanımı)
  - T5 (SentencePiece Unigram)
  - ALBERT (SentencePiece Unigram)
* Yöntemlerin Karşılaştırılması
  - Sözcük Çantası/TF-IDF: Ne Zaman Kullanılmalı?
  - Gömmeler: Ne Zaman Kullanılmalı?
  - Altsözcük Atomlarına Ayırma: Optimal Sözcük Dağarcığı Boyutu
* Değerlendirme Ölçütleri
  - İçsel Değerlendirme (Intrinsic): Kelime Benzerliği, Analoji
  - Dışsal Değerlendirme (Extrinsic): Alt Görev Performansı


# Kursa Kayıt
[Kursumuza ön kayıt yaptırmak için bu bağlantıyı kullanabilirsiniz.](https://us02web.zoom.us/meeting/register/A2N5yVVXStuiYN0tAXTeJw#/registration)


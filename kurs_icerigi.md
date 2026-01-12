# Online Üretici Yapay Zeka, Doğal Dil İşleme ve Büyük Dil Modelleri (350 saat)


Açıklama: Bu kursumuz üretici ağlar, doğal dil işleme ve büyük dil modellerini çeşitli düzeylerde kapsamlı bir biçimde ele almaktadır. Kursumuz yapay zeka alanında devrimsel bir etkiye yol açan bu teknolojilerin teorik altyapıları ve gerçekleştirimleri konusunda uygulamalı bilgiler vermeyi ve beceriler kazandırmayı hedeflemektedir. Bu alan çok hızlı ilerlediği için ve kursumuz uzun sürdüğü için kurs sırasında başka konu eklemeleri yapılabilecektirr. Aşağıdaki belirttiğimiz kurs konuları temel alınacak ve bu konular çeşitli derinliklerde ele alınacaktır.

* __Gözden Geçirme ve Hazırlık__
   * Yapay Zeka, Makine Öğrenmesi ve Veri Bilimi İçin Neden Python? 
   * Yapay Zeka, Makine Öğrenmesi ve Veri Biliminde Diğer Programlama Dillerinin Avantajları ve Dezavantajları
   * Python Programlama Dilinin Yeniden Gözden Geçirilmesi
   * Scipy, Numpy, Pandas ve Matplotlib Kütüphanelerinin Genel Tanıtımı
   * SciPy ve scikit-learn Kütüphanelerinin Genel Tanıtımı
   * Tensorflow, Keras ve PyTorch Kütüphanelerinin Genel Tanıtımı
   * HuggingFace ve Diğer Makine Öğrenmesi ve Doğal Dil İşleme Kütüphanelerinin Genel Tanıtımı
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
  * Atomlara Ayırmada Düzeyler
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
* __Klasik Doğal Dil İşleme Algoritmalarının Temel Düzeyde Gözden Geçirilmesi__
  * Naive Bayes Yöntemi
  * Lojistik Regresyon Yöntemi
  * Destek Vektör Makineleri (SVM) Yöntemi
  * Koşullu Rastgele Alanlar (CRF) Yöntemi
  * LDA (Latent Dirichlet Allocation) Yöntemi
  * LSA (Latent Semantic Analysis) Yöntemi
* __Modern Doğal Dil İşleme Yöntemlerine Hazırlık__
   * Yapay Sinir Ağı Nedir?
  * Yapay Sinir Ağlarının Tarihsel Gelişimi
  * Yapay Sinir Ağlarının Uygulama Alanları
  * Yapay Nöron Modeli
  * Yapay Bir Nöronun Python'da Sınıfsal Temsili
  * Aktivasyon Fonksiyonları ve Nöron Bağlantıları
  * Yapay Sinir Ağlarında Katmanlar
  * Yapay Sinir Ağlarında Yaygın Kullanılan Kütüphaneler ve Framework'ler
  * Yapay Sinir Ağlarında Veri Kümeleri (Eğitim Veri Kümesi, Sınama Veri Kümesi ve Test Veri Kümesi)
  * Keras Kütüphanesinin Kısa Tarihi
  * Keras Kütüphanesinin Kurulumu
  * Yapay Sinir Ağlarının Keras Kütüphanesi İle Yüksek Seviyeli Biçimde Oluşturulması
  * Girdi Katmanlarının, Ara Katmanların ve Çıktı Katmanlarının Oluşturulması
  * Loss Fonksiyonlarının Anlamı ve Çok Kullanılan Loss Fonksiyonları
  * Optimizasyon Algoritmalarının Anlamı ve Çok Kullanılan Optimizasyon Algoritmaları
  * Keras'ta Yapay Sinir Ağı Modelinin Eğitilmesi
  * Epoch ve Batch Kavramları
  * Sınama (Validation) İşleminin Anlamı
  * Test Veri Kümesi ile Yapay Sinir Ağı Modelinin Test Edilmesi
  * Keras Modelinin Dosyalarda Saklanması ve Geri Yüklenmesi
  * Geri Beslemeli Sinir Ağları (Recurrent Neural Networks)
  * Geri Beslemeli Sinir Ağlarında SimpleRNN, LSTM ve GRU Katmanları
  * Çift Yönlü LSTM Katmanları
  * Geri Beslemeli Sinir Ağlarının Kestirimde Kullanılmaları
  * Geri Beslemeli Sinir Ağlarına İlişkin Örnekler
  * Öğrenimin Aktarılması Sürecinin (Transfer Learning) Temelleri
  * Evrişim İşlemleri ve Evrişim İşlemlerinin Bağlamsal Anlamı
* __Dönüştürücüler (Tramsformers) ve Modern Doğal Dil İşlemeye Geçiş__
  * Temel Kavramlar ve Tarihsel Bağlam
    - Dizi İşleme Mimarileri ve Sınırlamaları
    - RNN (Recurrent Neural Networks) Temelleri
    - LSTM ve GRU Mimarileri
    - Vanishing/Exploding Gradient Problemi
    - Uzun Bağımlılıkların İşlenememesi
    - Paralel Hesaplama Sınırlamalar
  * Sequence-to-Sequence Modeller
    - Encoder-Decoder Yapısı
    - Klasik Attention Mekanizması
    - Bahdanau (Additive) ve Luong (Multiplicative) Dikkat
    - Hizalama (Alignment) Kavramı
  * Dönüştürücü Modeline Geçiş
    - Attention is All You Need" Makalesi İncelemesi
    - Geri Beslemenin (Recurrence)  Kaldırılması
    - Dönüştürücülerin Sağladığı Faydalar
  * Dikkat Mekanizması
    - Scaled Dot-Product Attention
    - Çok Başlı (Multi-Head) Dikkat
    - Dikkat Türleri
  * Dikkat Mekanizmasının Tensorflow/Keras ile Gerçekleştirimi
  * Dönüştürücü Mimarisi
    - Encoder Yapısı
    - Decoder Yapısı
    - Pozisyonel Encoding
    - Dönüştürücülerin Diğer Bileşenleri
    - Dönüştürücülerin Tensorflow/Keras İle Gerçekleştirimi
* __Önceden Eğitilmiş Modeller__
  * BERT (Bidirectional Encoder Representations from Transformers)
    - Mimari Detayları (Base ve Large)
    - Masked Language Modeling (MLM)
    - Next Sentence Prediction (NSP)
    - Önceden Eğitim (Pre-training) ve İnce Ayar (Fine-tuning) Mekanizmaları
  * GPT (Generative Pre-trained Transformer)
    - GPT-1, GPT-2, GPT-3 Evrimi
    - Tek Yönlü (Unidirectional) Decoder Mimarisi
    - Autoregressive Language Modeling
    - Zero-shot, Few-shot, One-shot Öğrenim
  * T5 (Text-to-Text Transfer Transformer)
    - Unified Text-to-Text Framework
    - C4 Dataset ve Pre-training Görevleri
    - Encoder-Decoder Mimarisi
  * Diğer Önemli Modeller
    - RoBERTa (Robustly Optimized BERT)
    - ALBERT (A Lite BERT)
    - ELECTRA (Efficiently Learning an Encoder)
    - DeBERTa (Decoding-enhanced BERT)
  * Model Karşılaştırmaları
    - Çift Yönlü (Bidirectional) ve Tek Yönlü (Unidirectional) Dİkkat
    - Encoder-only,  Decoder-only, Encoder-Decoder Modeller
    - Model Boyutları ve Performans İlişkisi
  * Hugging Face Transformers Kütüphanesinin Kullanımı
    - Model Hub Kullanımı
    - AutoModel ve AutoTokenizer
    - Pipeline API ile Hızlı Kullanım
    - İnce Ayar İşlemleri İşlemleri
    - Model Kaydetme ve Yükleme     
*__Büyük Dil Modelleri__
  * GPT ve Büyük Dil Modelleri
  * LLaMA Mimarisi
  * LLaMA Türevleri
  * Büyük Dil Modellerinin Temelleri ve Ölçeklendirme
    - Büyük Dil Modellerine Giriş
    - Model Ölçeklendirme Stratejileri
    - LLM Mimarisi Evrimleri
    - Atomlarına Ayırma Stratejileri
  * Büyük Dil Modelleri ve Önceden Eğitim 
    - Autoregressive Language Modeling
    - Önceden Eğitilmiş Veri Kümeleri
    - Pre-training Optimizasyonu
  * Dağıtık Biçimde Eğitimin Yapılış Biçimleri
  * Stabilite
  * Büyük Dil Modellerinde İnce Ayar (Fine Tuning) ve Adaptasyon
    - Denetimli İnce Ayar Mekanizmaları (SFT)
    - Parameter-Efficient Fine-tuning (PEFT)
    - Pekiştirmeli Öğrenmeden Faydalanma
    - Direct Preference Optimization (DPO)
    - Diğer Teknikler
  * Prompt İşlemleri ve Bağlam İçinde Öğrenme
    - Zero-shot Prompt'lama
    - Few-shot Prompt'lama
    - Negative Prompt'lama
    - Prompt Template Tasarımı
  * İleri Prompting Teknikleri
  * Büyük Dil Modellerinde Değerlendirme ve Benchmark İşlemleri
    - Geleneksel NLP Metrikleri
    - Büyük Dil Modellerine Özgü Benchmark Testleri
    - Akıl Yürütme ve Yetenek Testleri
    - İnsanlar Tarafından Değerlendirme
  * Büyük Dil Modellerindeki Etik ve Sorumluluk Gerektiren Sornlar
    - Büyük Dil Modellerinde Yanlılık ve Tarafsızlık
    - Zararlı İçerikler
    - Halüsinasyon Olgusu
    - Özel Bilgilerin Güvenliği
* __Açık Kaynak Kodlu Büyük Dil Modelleri__
  * Büyük Dil Uygulama Platformları Hakkında Bilgiler
    - LangChain Architecture
    - LlamaIndex (RAG Framework)
    - Semantic Kernel (Microsoft)
    - AutoGen (Multi-Agent Framework)
  * Vektör Veritabanları
* __Küçük Dİl Modelleri (Small Language Models)__
  * Küçük Dil Modellerine Giriş
  * Büyük Dil Modelleriyle Küçük Dil Modellerinin Karşılaştırılması
  * Modern Küçük Dil Modelleri Örnekleri ve Mimarileri
  * TinyLLaMA
  * MiniLM
  * DistilBERT
  * Gemma (Küçük Boyutlar)
  * Mobil Cihazlara Uygun Küçük Dil Modelleri
  * Küçük Dil Modellerinin Eğitimi ve Türetimi Teknikleri
  * Küçük Dil Modellerinin Kullanım Senaryoları ve Deployment
*__Retrieval-Augmented Generation (RAG) Sistemleri__
  * RAG Temelleri
    - RAG Neye Yarıyor?
    - Büyük Dil Modellerinde Halüsinasyon Problemi
    - Dışsal Bilgi Kaynaklarının Entegre Edilmesi
    - RAG ile İnce Ayar Mekanizmalarının Karşılaştırması
  * Retrieval Stratejileri
    - Dense Retrieval (DPR, ColBERT)
    - Sparse Retrieval (BM25, TF-IDF)
    - Hybrid Search
    - Gömülü Modelleri
  * Vector Veritabanarı
    - Pinecone, Weaviate, ChromaDB, Qdrant
    - Index Tipleri (HNSW, IVF)
    - Similarity Search
  * Doküman İşleme
    - Chunking Stratejileri
    - Metadata Extraction
    - Document Loaders
  * Re-ranking ve Filtering
    - Cross-encoder Re-rankers
    - Diversity ve MMR
  * Advanced RAG
    - Self-RAG
    - Düzeltici (Corrective) RAG (CRAG)
    - Ajanlı RAG
  * RAG Sistemlerinin Değerlendirilmesi
    - Retrieval Metrics
    - Generation Quality
- *__Yapay Zeka Ajanları ve Ajan İletişim Protokolleri__
  * Yapay Zeka Ajanlarının Temelleri
  * Ajan Mimarisi
  * Ajan Türleri
  * Büyük Dil Modelleri İçin Ajanlar
  * Ajan Protokolleri
  * MCP (Model Context Protocol)
  * MCP Kaynakları
  * MCP Araçları
  * MCP İle Kodlama İşlemleri
  * OpenAI Fonksiyon Çağırma İşlemleri
  * Agent Protocol Standardı (e2b)
  * LangChain Tool Protokolü
  * Protokollerin Karşılaştırılması
  * Çok Ajanlı Sistemler
    - Çok Ajanlı Mimariler
    - Ajan Rolleri ve ÖZellikleri
    - Ajan Haberleşmeleri
    - Çok Ajanları Framework'lere Genel Bakış
  * Yaygın Biçimde Kullanılan Ajan Framework'lerine Bakış
    - LangGraph
    - LangChain Ajanları
    - AutoGen
    - Semantic Kernel
    - Ajanlar İçin Bulut Masaüstü Çözümleri
      - Amazon AWS
      - Microsoft Azure
      - Google Cloud Platform
  * Kod Üretimi (Kod Generation) Ajanları
    - Kod Yazım Ajanları
    - Kod Çalıştırma Ortamları
    - Kod Ajanları İçin MCP
    - AutoGPT ve Türevleri
    - Yazılım Geliştirme Ajanları
  * Özelleştirilmiş Ajanlar
    - Araştırma Ajanları
    - Veri Analizi Ajanları
    - Müşteri Hizmetleri Ajanları
    - Kişisel Yardım Ajanları
  * Ajanların Değerlendirilmesi ve Test Edilmesi
    - Performans Metrikleri
    - Kalite Metrikleri
    - Benchmark Testleri
    - Test ve Debug Stratejileri
  * Ajanların Konuşlandırılması (Deployment)
    - Ajan Konuşlandırma Mimarileri
    - Ölçekleme Stratejileri
    - İzleme ve Gözlemleme
    - Maliyet Unsurlarını Gözden Geçirme
* __Büyük Dil Modelleribi Sıfırdan Eğitme__
  * Veri Hazırlama
  * Mini Büyük Dil Modeli Mimarisi
  * Eğitim Süreci
  * Değerlendirme
  * Konuşlandırma (Deployment) Yöntemlerine Toplu Bakış
* __Görüntüsel İşlemler Yapan üretici Ağlar__
    * GAN Temelleri
        - Generator / Discriminator
        - Min–max oyun teorisi
        - Loss fonksiyonları
    * Görüntü Tabanlı GAN Mimarileri
    * Görsel Üretim Uygulamaları
    * GAN ve Diffusion
    


    
      
    
   




# Kursa Kayıt
[Kursumuza ön kayıt yaptırmak için bu bağlantıyı kullanabilirsiniz.](https://us02web.zoom.us/meeting/register/A2N5yVVXStuiYN0tAXTeJw#/registration)


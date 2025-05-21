# Теоретические вопросы:
## 1. **Сформулируйте теорему Цыбенко. Что эта теорема показывает на практике.**
см. [яндекс учебник, 5.2](https://education.yandex.ru/handbook/ml/article/pervoe-znakomstvo-s-polnosvyaznymi-nejrosetyami)
## 2.**Функции активации: какие бывают**, в каких архитектурах используются, какие интерпретации им можно дать.
см. [яндекс учебник, 5.2](https://education.yandex.ru/handbook/ml/article/pervoe-znakomstvo-s-polnosvyaznymi-nejrosetyami)
#TODO: в каких архитектурах используются?
## 3.**Метод back propagation**. Как устроены, какая вычислительная сложность и сложность по памяти. Activation checkpointing. Обучение в условиях ограниченной памяти.
см. [яндекс учебник, 5.3](https://education.yandex.ru/handbook/ml/article/metod-obratnogo-rasprostraneniya-oshibki)
#TODO граф вычислений, какая вычислительная сложность и сложность по памяти. Activation checkpointing. Обучение в условиях ограниченной памяти.
## 4. Стохастический градиентный спуск. Осцилляция SGD в регионе неопределенности, условия сходимости метода. 
см. [яндекс учебник 2.1 (Стохастический градиентный спуск(SGD))](https://education.yandex.ru/handbook/ml/article/linear-models#linejnaya-regressiya-i-metod-naimenshih-kvadratov-mnk), [яндекс учебник 14.4 (Сходимость SGD)](https://education.yandex.ru/handbook/ml/article/shodimost-sgd), [лекция 3, слайд 11 (сходимость SGD, непонятно)](https://docs.google.com/presentation/d/1RFskwhHeI82DGlVhbYB6gcUmd5JRL3DRZyHoevRUuAI/edit?slide=id.g3023b418fcb_0_57#slide=id.g3023b418fcb_0_57)
## 5. Метод моментов Поляка. Чем отличается имплементация в Pytorch от оригинального метода. Метод Adam. Почему он устроен именно так.  
## 6. Sharpness Aware Minimization. LION, Sophia.  
## 7. Регуляризация. Weight decay, max-norm constraint, дропаут, обрезка градиентов (gradient clipping) и ранняя остановка. В каких случаях применяется. SGDW, AdamW. Батч-нормализация. Как она работает во время обучения и во время инференса. 
## 8. Инициализация весов. Какие бывают способы инициализации и каким условиям они должны удовлетворять.
[см. яндекс учебник 5.4 (почти всё)](https://education.yandex.ru/handbook/ml/article/tonkosti-obucheniya#inicializiruem-pravilno), [см лекция 4, слайд 20 (MagicInit (непонятно))](https://docs.google.com/presentation/d/1M0cuCzLETPdqTvahRSJOfJhysmL4RsBriC-OqHoL5F8/edit?slide=id.p18#slide=id.p18)
## 9. Что такое свертка. Какое в ней число FLOPS, как устроен тензор весов. Какие в ней бывают параметры. Как устроены простые сверточные архитектуры. 
## 10. Устройство VGG, Inception GoogLeNet, Depthwise convolution.
[см. документ (так себе)](https://docs.google.com/document/u/0/d/1PeahfK17HZC6vJe7jDnsmMM0xhn9w3hIA2ZOaCIZLrE/mobilebasic), [см. лекция 6](https://docs.google.com/presentation/d/18o2oks8xbhSWYYxJpukYe16C-boiKV_3OQ-yDqMV9ys/edit?slide=id.p1#slide=id.p1) [см. яндекс учебник 6.1 (Бонус 1 (обзорно))](https://education.yandex.ru/handbook/ml/article/svyortochnye-nejroseti#bonus-1-znakovye-arhitektury-v-mire-svyortochnyh-nejronnyh-setej-dlya-zadachi-klassifikacii-izobrazhenij)
## 11. Как выглядят bottleneck в Resnet и MobileNet v2. Почему они устроены именно так. SE-блоки и Efficient Net. Efficient Net v2.
## 12. Постановка задачи детекции. Какие в ней могут быть метрики. Как они вычисляются. Зачем нужен NMS. Как выглядит функция потерь.
## 13. Two-stage detectors: семейство R-CNN.
## 14. One-stage detectors: семейство Yolo, SSD.
## 15. One-stage anchor free detectors:  CornerNet, FCOS, CenterNet, CentripetalNet
## 16. Какие бывают задачи сегментации. Какие бывают метрики для сегментации. Функции потерь для сегментации.
## 17. Как можно делать повышение размерности: ConvTranspose, Upscale, Unpooling. Как работают эти способы. Архитектура Unet и DeepLab. Dilated convolutions, как выглядит ASPP.
## 18. Методы векторизации: bag of words, tf-idf, word2vec. Как они работают и где используются.
## 19. Методы токенизации: по словам, по символам, BPE.  
## 20. RNN, Bidirectional RNN. Разворачивание RNN. Обучение RNN. Проблемы затухающих и взрывающихся градиентов. LSTM, GRU.  
## 21. Механизм внимания для RNN. Механизм внимания в трансформерных архитектурах. Как они выглядят, как их можно интерпретировать.  
## 22. Кодировщик в трансформере. Self-attention. Multihead self-attention. Positional encoding. Декодировщик в трансформере. Masked self-attention. Обучение трансформера и инференс.  
## 23. Энкодерные модели: BERT, ALBERT, ViT, CaiT, Swin, PVT, Linformer.  
## 24. Декодерные модели: задача языкового моделирования, GPT, LLAMA. Перплексия и методы генерации. Особенности инференса.  
## 25. Обучение LLM. Какие стадии бывают. PEFT. Как и когда использовать различные методы PEFT.  
## 26. SSM. Линейные трансформеры, Гиена, Mamba, RWKV, Retention.  
## 27. Дистилляция. Что такое, где используется, какие есть методы.  
## 28. GAN. Архитектура, функция потерь. Adversarial Loss, Gradient penalty. Spectral Normalization.  
## 29. Оптимизация гиперпараметров как задача дискретной оптимизации: случайный поиск, поиск по сетке, генетические алгоритмы, стохастические алгоритмы.  
## 30. Neural Architecture Search: дифференцируемый и не дифференцируемый. DARTS, Proxyless NAS, Once For All.  
## 31. Распределенное обучение Data Parallel, Tensor Parallel, Pipeline Parallel. Как устроены, стратегии их применения. Недостатки и достоинства.  
## 32. Устройство CPU, GPU. Парадигма CUDA, Triton. Блочные алгоритмы вычисления матричных умножений и roofline model, micro и macro kernels. Вычисление на систолических массивах и тензорных ядрах.
## 33. Mixed Precision Training. Как устроено обучение в fp16. bfp16, fp8. Что такое micro scaling data types.  
## 34. Quantization. Что такое, как работает, какие бывают виды. Работа с выбросами в сверточных сетях и в трансформерах. Обучаемые пороги.  
## 35. Прунинг. Гранулярность, зависимость от HW. Критерии прунинга. Стратегии работы с прунингом (когда прунить, как обучать)

# Практические задачи:

## 1. Можно ли использовать L(x)=0 в качестве критерия остановки обучения. Почему?
## 2. Пусть у нас есть тензор A\[B, C, H, W\]. Как мы можем взять из него произвольный элемент, каждый второй элемент по оси C каждый второй элемент по оси H? Ответьте на этот вопрос для случаев, когда тензор лежит в памяти в виде одномерной row-order contiguous строки, в виде channelslast представления, в виде column-order contigous строки
## 3. Что такое leaf tensor в pytorch? Какими способами его можно создать? Какими способами можно создать non-leaf tensor? Из каких тензоров можно сформировать вычислительный граф?  
## 4. Чем fx.symbolic\_trace() отличается torch.compile()? Какие графы получены в процессе или в результате их выполнения? Как эти графы представлены, какое из представлений более низкоуровневое и почему?  
## 5. При вычислении производной в узле с помощью обратного распространения ошибки часто нужны дополнительные тензоры. Какой механизм в pytorch предназначен для работы с ними? Такие тензоры сохраняются автоматически или нам необходимо их сохранить вручную? Какие тензоры сохраняются для операций X \* Y, X @ Y, e^X, X / Y, sqrt(X), ReLU(X)?  
## 6. Как можно вычислить гессиан по графу, вычисляющему первые производные? Этот вопрос можно разбить на два: как можно вычислить вторые производные и как \- частные.  
## 7. Как можно представить свертку со страйдом с помощью обычной свертки?  
## 8. Каким должен быть размер паддинга, чтобы изображение после свертки 3х3 не поменяло свой размер? А после свертки 4х4?  
## 9. Как представить свертку в виде матричного умножения? А матричное умножение в виде свертки?  
## 10. Какое число FLOPs в свертке?  
## 11. Как реализовать билинейную интерполяцию через операцию свертки?  
## 12. Какая вычислительная сложность и сложность по памяти у self-attention.

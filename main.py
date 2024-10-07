# 1. 데이터셋 객체 생성
dataset = Dataset(ood=True)

# 2. 임베딩 프로세서 생성
emb = GensimEmbedder(model=embed.FastText())

# 3. 의도(Intent) 분류기 생성
clf = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),                  
    loss=CenterLoss(dataset.intent_dict)                    
)

# 4. 개체명(Named Entity) 인식기 생성                                                     
rcn = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

# 5. 딥러닝 챗봇 RESTful API 학습 & 빌드
kochat = KochatApi(
    dataset=dataset, 
    embed_processor=(emb, True), 
    intent_classifier=(clf, True),
    entity_recognizer=(rcn, True), 
    scenarios=[
        weather, dust, travel, restaurant
    ]
)

# 6. View 소스파일과 연결                                                                                                        
@kochat.app.route('/')
def index():
    return render_template("index.html")

# 7. 챗봇 애플리케이션 서버 가동                                                          
if __name__ == '__main__':
    kochat.app.template_folder = kochat.root_dir + 'templates'
    kochat.app.static_folder = kochat.root_dir + 'static'
    kochat.app.run(port=8080, host='0.0.0.0')

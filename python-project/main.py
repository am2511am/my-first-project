from sklearn.tree import DecisionTreeClassifier
ข้อมูล  = [[1,40], [2,45], [3,55], [5,65], [6,80], [8,90]]  # Feature
ผลสอบ = ["Fail","Fail","Pass","Pass","Pass","Pass"]  # Label
model = DecisionTreeClassifier()

model.fit(ข้อมูล,ผลสอบ)  # Training

result = model.predict([[4,60]])
print("ชั่วโมงเรียน:", 4)
print("คะแนนสอบ:", 60)
print("ผลการทํานาย:", result[0])
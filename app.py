from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# โหลดและทำความสะอาดข้อมูลจากไฟล์ .csv
data = pd.read_csv('data/Mobile-Price.csv')
data_cleaned = data.drop(columns=['Ratings'])  # ลบคอลัมน์ที่ไม่ต้องการ
X = data_cleaned[['RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']]
y = data_cleaned['Price']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ฝึกโมเดลใหม่ทุกครั้งเมื่อรันโค้ด
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# แสดงข้อมูลการฝึก (100 แถวแรก)
training_data = data_cleaned.head(100)

@app.route('/')
def home():
    # แปลงข้อมูลทำนายเป็น list
    predictions = model.predict(X_test).tolist()
    # แปลงข้อมูลตารางเพื่อส่งไปแสดงผลในหน้าเว็บ
    tables = [training_data.to_html(classes='data', header="true")]
    titles = training_data.columns.values
    return render_template('index.html', predictions=predictions, tables=tables, titles=titles)

@app.route('/predict', methods=['POST'])
def predict():
    # รับค่าที่ส่งมาจากฟอร์ม
    try:
        ram = float(request.form['RAM'])
        rom = float(request.form['ROM'])
        mobile_size = float(request.form['Mobile_Size'])
        primary_cam = float(request.form['Primary_Cam'])
        selfi_cam = float(request.form['Selfi_Cam'])
        battery_power = float(request.form['Battery_Power'])

        # สร้าง DataFrame สำหรับการทำนาย
        input_data = pd.DataFrame([[ram, rom, mobile_size, primary_cam, selfi_cam, battery_power]],
                                  columns=['RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power'])

        # ทำนายราคาจากข้อมูลที่ได้รับ
        predicted_price = model.predict(input_data)[0]

        # ส่งผลลัพธ์การทำนายกลับในรูปแบบ JSON
        return jsonify({'prediction': predicted_price})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

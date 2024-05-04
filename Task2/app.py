from flask import Flask, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import io
from datetime import datetime
import json

app = Flask(__name__)

with open('loans_data.json', 'r') as json_file:
    data = json.load(json_file)
df = pd.DataFrame(data['loans'])

df['start_date'] = pd.to_datetime(df['start_date'])
df['maturity_date'] = pd.to_datetime(df['maturity_date'])
df.drop_duplicates(subset='loan_id', keep='first', inplace=True)
df = df[df['start_date'] < df['maturity_date']]

current_year = pd.Timestamp.now().year
df['year'] = df['start_date'].dt.year
df['month'] = df['start_date'].dt.month
df_current_year = df[df['year'] == current_year]

monthly_loan_counts = df_current_year.groupby('month')['status'].value_counts().unstack(fill_value=0)
new_loans_by_month = monthly_loan_counts['Active']
closed_loans_by_month = monthly_loan_counts['Completed']

@app.route('/loan_metrics', methods=['GET'])
def loan_metrics():
    
    loan_metrics_data = {
        'total_active_loans': int(df[df['status'] == 'Active'].shape[0]),
        'total_completed_loans': int(df[df['status'] == 'Completed'].shape[0]),
        'monthly_amount_financed_current_year': df_current_year.groupby(df_current_year['start_date'].dt.month)['loan_amount'].sum().astype(float).to_dict(),
        'total_outstanding_amount_with_interest': float((df[df['status'] == 'Active']['loan_amount'] * (1 + df[df['status'] == 'Active']['profit_percentage'] / 100) ** ((datetime.now() - df[df['status'] == 'Active']['start_date']).dt.days / 365)).sum())
    }
    
    active_loans = df[df['status'] == 'Active']
    
    def calculate_outstanding_amount(row):
        remaining_days = (row['maturity_date'] - datetime.now()).days
        
        remaining_principal_with_interest = (row['loan_amount'] *
                                             (1 + row['profit_percentage'] / 100) **
                                             (remaining_days / 365))
        return float(remaining_principal_with_interest)
    
    active_loans['outstanding_amount_with_interest'] = active_loans.apply(calculate_outstanding_amount, axis=1)
    
    total_outstanding_amount = float(active_loans['outstanding_amount_with_interest'].sum())
    
    loan_metrics_data['total_outstanding_amount_with_interest'] = total_outstanding_amount
    
    
    month_wise_loan_data = {}
    for month in range(1, 13):
        month_name = pd.Timestamp(year=current_year, month=month, day=1).strftime('%B')
        new_loans_count = int(new_loans_by_month.get(month, 0))
        closed_loans_count = int(closed_loans_by_month.get(month, 0))
        month_wise_loan_data[month_name] = {
            'new_loans': new_loans_count,
            'closed_loans': closed_loans_count
        }
    
    loan_metrics_data['month_wise_loan_data_current_year'] = month_wise_loan_data
    
    json_loan_metrics_data = {key: (value if isinstance(value, (int, float, str, list, dict)) else str(value)) for key, value in loan_metrics_data.items()}
    return jsonify(json_loan_metrics_data)


@app.route('/generate_report', methods=['GET'])
def generate_report():
    
    month_wise_loan_data = {}
    current_year = 2024 
    for month in range(1, 13):
        month_name = pd.Timestamp(year=current_year, month=month, day=1).strftime('%B')
        new_loans_count = int(new_loans_by_month.get(month, 0))
        closed_loans_count = int(closed_loans_by_month.get(month, 0))
        month_wise_loan_data[month_name] = {
            'new_loans': new_loans_count,
            'closed_loans': closed_loans_count
        }

    monthly_amount_financed = df_current_year.groupby('month')['loan_amount'].sum()

    print("Month-wise Total Amount Financed for Loans (Current Year):")
    for month in range(1, 13): 
        total_amount = monthly_amount_financed.get(month, 0)
        month_name = pd.Timestamp(year=current_year, month=month, day=1).strftime('%B')
        print(f"{month_name}: ${total_amount:.2f}")

    plt.figure(figsize=(6, 3))
    monthly_amount_financed.plot(kind='bar', color='skyblue')
    plt.title('Month-wise Total Amount Financed for Loans (Current Year)', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('monthly_amount_financed_bar_chart.png')

    plt.figure(figsize=(8, 6))
    df['status'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title('Loan Status Distribution')
    plt.savefig('loan_status_pie_chart.png')

    plt.figure(figsize=(10, 8))
    months = list(month_wise_loan_data.keys())
    new_loans = [month_wise_loan_data[month]['new_loans'] for month in months]
    closed_loans = [month_wise_loan_data[month]['closed_loans'] for month in months]
    width = 0.35
    plt.bar(months, new_loans, width, label='New Loans')
    plt.bar(months, closed_loans, width, label='Closed Loans', bottom=new_loans)
    plt.xlabel('Month')
    plt.ylabel('Number of Loans')
    plt.title('Month-wise Loan Data')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('month_wise_loan_data_bar_chart.png')

    buffer = io.BytesIO()
    pdf_canvas = plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.imshow(plt.imread('loan_status_pie_chart.png'))
    plt.axis('off')
    plt.title('Loan Status Distribution (Pie Chart)')

    plt.subplot(3, 1, 2)
    plt.imshow(plt.imread('month_wise_loan_data_bar_chart.png'))
    plt.axis('off')
    plt.title('Month-wise Loan Data (Bar Chart)')

    plt.subplot(3, 1, 3)
    plt.imshow(plt.imread('monthly_amount_financed_bar_chart.png'))
    plt.axis('off')
    plt.title('Month-wise Total Amount Financed for Loans')

    plt.tight_layout()
    plt.savefig(buffer, format='pdf')
    buffer.seek(0)
    
    return send_file(buffer, mimetype='application/pdf', as_attachment=True, download_name='loan_report.pdf')

if __name__ == '__main__':
    app.run(port=9090, debug=True)
# from flask import Flask, request, jsonify, render_template_string
# import joblib
# import pandas as pd
# import numpy as np

# app = Flask(__name__)

# # 1. Load the serialized pipeline artifact
# MODEL_PATH = 'demand_prediction_pipeline.pkl'
# pipeline = joblib.load(MODEL_PATH)

# # Categorical mapping dictionaries matching Step 7
# SEASONALITY_MAP = {'Fall': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}

# # Modern UI Template using Tailwind CSS and Inter typography
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>SupplyPulse - AI Demand Forecaster</title>
#     <!-- Tailwind CSS CDN -->
#     <script src="https://cdn.tailwindcss.com"></script>
#     <link rel="preconnect" href="https://fonts.googleapis.com">
#     <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
#     <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
#     <style>
#         body { font-family: 'Plus Jakarta Sans', sans-serif; }
#     </style>
# </head>
# <body class="bg-slate-900 text-slate-100 min-h-screen py-10 px-4 antialiased">
#     <div class="max-w-4xl mx-auto">
        
#         <!-- Header -->
#         <div class="text-center mb-10">
#             <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold uppercase tracking-wider mb-3">
#                 <span class="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
#                 Machine Learning Inference Engine
#             </div>
#             <h1 class="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">SupplyPulse - Retail Demand Forecaster</h1>
#             <p class="text-slate-400 text-sm sm:text-base mt-2 max-w-xl mx-auto">
#                 Optimize stock levels, reduce stockouts, and predict daily demand units.
#             </p>
#         </div>

#         <!-- Main Card -->
#         <div class="bg-slate-800/80 border border-slate-700/80 backdrop-blur-xl rounded-2xl shadow-2xl p-6 sm:p-10">
            
#             <form method="POST" action="/predict" class="space-y-8">
                
#                 <!-- Section 1: Product & Environment -->
#                 <div>
#                     <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
#                         <svg class="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
#                         Catalog & Context
#                     </h2>
#                     <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        
#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Category</label>
#                             <select name="category" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                                 {% for opt in ['Electronics', 'Clothing', 'Furniture', 'Groceries', 'Toys'] %}
#                                 <option value="{{ opt }}" {% if form_data.category == opt %}selected{% endif %}>{{ opt }}</option>
#                                 {% endfor %}
#                             </select>
#                         </div>

#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Region</label>
#                             <select name="region" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                                 {% for opt in ['North', 'South', 'East', 'West'] %}
#                                 <option value="{{ opt }}" {% if form_data.region == opt %}selected{% endif %}>{{ opt }}</option>
#                                 {% endfor %}
#                             </select>
#                         </div>

#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Seasonality</label>
#                             <select name="seasonality" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                                 {% for opt in ['Winter', 'Spring', 'Summer', 'Fall'] %}
#                                 <option value="{{ opt }}" {% if form_data.seasonality == opt %}selected{% endif %}>{{ opt }}</option>
#                                 {% endfor %}
#                             </select>
#                         </div>

#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Weather</label>
#                             <select name="weather" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                                 {% for opt in ['Sunny', 'Rainy', 'Snowy', 'Cloudy'] %}
#                                 <option value="{{ opt }}" {% if form_data.weather == opt %}selected{% endif %}>{{ opt }}</option>
#                                 {% endfor %}
#                             </select>
#                         </div>

#                     </div>
#                 </div>

#                 <!-- Section 2: Pricing & Competition -->
#                 <div>
#                     <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
#                         <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
#                         Pricing & Elasticity
#                     </h2>
#                     <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        
#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Our Price ($)</label>
#                             <input type="number" step="0.01" name="price" value="{{ form_data.price }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                         </div>

#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Competitor Price ($)</label>
#                             <input type="number" step="0.01" name="competitor_pricing" value="{{ form_data.competitor_pricing }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                         </div>

#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Discount Rate (%)</label>
#                             <input type="number" step="1" name="discount" value="{{ form_data.discount }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                         </div>

#                     </div>
#                 </div>

#                 <!-- Section 3: Inventory, Orders & Market Conditions -->
#                 <div>
#                     <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
#                         <svg class="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
#                         Supply & Market Signals
#                     </h2>
#                     <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        
#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Current Stock Level</label>
#                             <input type="number" name="inventory_level" value="{{ form_data.inventory_level }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                         </div>

#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Replenishment Units</label>
#                             <input type="number" name="units_ordered" value="{{ form_data.units_ordered }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                         </div>

#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Promotional Campaign</label>
#                             <select name="promotion" class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                                 <option value="1" {% if form_data.promotion|string == '1' %}selected{% endif %}>Active Promo</option>
#                                 <option value="0" {% if form_data.promotion|string == '0' %}selected{% endif %}>Standard</option>
#                             </select>
#                         </div>

#                         <div>
#                             <label class="block text-xs font-medium text-slate-300 mb-1.5">Epidemic / Disruption</label>
#                             <select name="epidemic" class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
#                                 <option value="0" {% if form_data.epidemic|string == '0' %}selected{% endif %}>Normal State</option>
#                                 <option value="1" {% if form_data.epidemic|string == '1' %}selected{% endif %}>Crisis / Wave</option>
#                             </select>
#                         </div>

#                     </div>
#                 </div>

#                 <!-- Submit Button -->
#                 <div>
#                     <button type="submit" class="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-sm tracking-wide shadow-lg shadow-blue-500/25 active:scale-[0.99] transition duration-150 flex items-center justify-center gap-2">
#                         <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
#                         Compute Forecasted Demand
#                     </button>
#                 </div>

#             </form>

#             <!-- Results Banner -->
#             {% if prediction is not none %}
#             <div class="mt-8 pt-8 border-t border-slate-700/60">
#                 <div class="bg-gradient-to-br from-blue-900/40 via-indigo-900/20 to-slate-900/80 border border-blue-500/30 rounded-2xl p-6 text-center shadow-inner relative overflow-hidden">
#                     <div class="absolute -right-10 -bottom-10 w-40 h-40 bg-blue-500/10 rounded-full blur-2xl pointer-events-none"></div>
#                     <span class="text-xs font-semibold text-blue-400 uppercase tracking-widest block mb-1">Inference Complete</span>
#                     <div class="text-4xl sm:text-5xl font-black text-white tracking-tight my-2">
#                         {{ prediction }} <span class="text-xl sm:text-2xl font-semibold text-blue-300">Units</span>
#                     </div>
#                     <p class="text-xs text-slate-400 mt-2">
#                         Predicted daily consumer purchase volume based on selected pricing, promotion, and seasonal indicators.
#                     </p>
#                 </div>
#             </div>
#             {% endif %}

#         </div>

#         <!-- Footer -->
#         <footer class="mt-8 text-center text-xs text-slate-500">
#             Designed & Engineered by <span class="text-slate-300 font-medium">Pranav T. Varshan</span> &bull; End-to-End Machine Learning System
#         </footer>

#     </div>
# </body>
# </html>
# """

# # Default initial form state
# DEFAULT_FORM = {
#     'category': 'Electronics',
#     'region': 'North',
#     'seasonality': 'Winter',
#     'weather': 'Sunny',
#     'price': '75.00',
#     'competitor_pricing': '80.00',
#     'discount': '10',
#     'inventory_level': '200',
#     'units_ordered': '150',
#     'promotion': '0',
#     'epidemic': '0'
# }

# def prepare_features(data_dict):
#     """Transforms raw dictionary inputs into the exact 25 model feature columns."""
#     price = float(data_dict['price'])
#     discount = float(data_dict['discount'])
#     competitor_price = float(data_dict['competitor_pricing'])
#     promotion = int(data_dict['promotion'])
#     epidemic = int(data_dict['epidemic'])
#     inv_level = float(data_dict['inventory_level'])
#     units_ordered = float(data_dict['units_ordered'])
    
#     category = data_dict['category']
#     region = data_dict['region']
#     weather = data_dict['weather']
#     season = data_dict['seasonality']
    
#     # Feature transformations
#     units_ordered_log = np.log1p(max(0, units_ordered))
#     inventory_level_sqrt = np.sqrt(max(0, inv_level))
#     price_diff = competitor_price - price
#     discount_amount = price * (discount / 100.0)
#     price_ratio = price / (competitor_price + 1e-5)
#     seasonality_encoded = SEASONALITY_MAP.get(season, 0)
    
#     # Calendar approximations for incoming query
#     month = 6
#     day_of_week = 2
#     quarter = 2
#     is_weekend = 0

#     features = {
#         'Price': price,
#         'Discount': discount,
#         'Promotion': promotion,
#         'Competitor Pricing': competitor_price,
#         'Epidemic': epidemic,
#         'Units_Ordered_log': units_ordered_log,
#         'Inventory_Level_sqrt': inventory_level_sqrt,
#         'Month': month,
#         'DayOfWeek': day_of_week,
#         'Quarter': quarter,
#         'Is_Weekend': is_weekend,
#         'Price_Diff': price_diff,
#         'Discount_Amount': discount_amount,
#         'Price_Ratio': price_ratio,
#         'Seasonality_Encoded': seasonality_encoded,
#         'Category_Electronics': 1 if category == 'Electronics' else 0,
#         'Category_Furniture': 1 if category == 'Furniture' else 0,
#         'Category_Groceries': 1 if category == 'Groceries' else 0,
#         'Category_Toys': 1 if category == 'Toys' else 0,
#         'Region_North': 1 if region == 'North' else 0,
#         'Region_South': 1 if region == 'South' else 0,
#         'Region_West': 1 if region == 'West' else 0,
#         'Weather Condition_Rainy': 1 if weather == 'Rainy' else 0,
#         'Weather Condition_Snowy': 1 if weather == 'Snowy' else 0,
#         'Weather Condition_Sunny': 1 if weather == 'Sunny' else 0
#     }
#     return pd.DataFrame([features])

# @app.route('/', methods=['GET'])
# def home():
#     return render_template_string(HTML_TEMPLATE, prediction=None, form_data=DEFAULT_FORM)

# @app.route('/predict', methods=['POST'])
# def predict():
#     input_data = request.form
#     feature_df = prepare_features(input_data)
#     predicted_val = pipeline.predict(feature_df)[0]
#     final_output = max(0, int(round(predicted_val)))
#     return render_template_string(HTML_TEMPLATE, prediction=final_output, form_data=input_data)

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     payload = request.get_json()
#     feature_df = prepare_features(payload)
#     pred = pipeline.predict(feature_df)[0]
#     return jsonify({
#         'status': 'success',
#         'predicted_demand_units': max(0, int(round(pred)))
#     })

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)


from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load the serialized pipeline artifact
MODEL_PATH = 'demand_prediction_pipeline.pkl'
pipeline = joblib.load(MODEL_PATH)

# Categorical mapping dictionaries matching Step 7
SEASONALITY_MAP = {'Fall': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}

# Modern UI Template with Blank Placeholders and Reset Action
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SupplyPulse - AI Demand Forecaster</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen py-10 px-4 antialiased">
    <div class="max-w-4xl mx-auto">
        
        <!-- Header -->
        <div class="text-center mb-10">
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold uppercase tracking-wider mb-3">
                <span class="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
                Machine Learning Inference Engine
            </div>
            <h1 class="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">SupplyPulse - Retail Demand Forecaster</h1>
            <p class="text-slate-400 text-sm sm:text-base mt-2 max-w-xl mx-auto">
                Optimize stock levels, reduce stockouts, and predict daily demand units.
            </p>
        </div>

        <!-- Main Card -->
        <div class="bg-slate-800/80 border border-slate-700/80 backdrop-blur-xl rounded-2xl shadow-2xl p-6 sm:p-10">
            
            <form method="POST" action="/predict" class="space-y-8">
                
                <!-- Section 1: Product & Environment -->
                <div>
                    <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                        <svg class="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                        Catalog & Context
                    </h2>
                    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        
                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Category</label>
                            <select name="category" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                                <option value="" disabled {% if not form_data.category %}selected{% endif %}>Select Category</option>
                                {% for opt in ['Electronics', 'Clothing', 'Furniture', 'Groceries', 'Toys'] %}
                                <option value="{{ opt }}" {% if form_data.category == opt %}selected{% endif %}>{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Region</label>
                            <select name="region" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                                <option value="" disabled {% if not form_data.region %}selected{% endif %}>Select Region</option>
                                {% for opt in ['North', 'South', 'East', 'West'] %}
                                <option value="{{ opt }}" {% if form_data.region == opt %}selected{% endif %}>{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Seasonality</label>
                            <select name="seasonality" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                                <option value="" disabled {% if not form_data.seasonality %}selected{% endif %}>Select Season</option>
                                {% for opt in ['Winter', 'Spring', 'Summer', 'Fall'] %}
                                <option value="{{ opt }}" {% if form_data.seasonality == opt %}selected{% endif %}>{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Weather</label>
                            <select name="weather" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                                <option value="" disabled {% if not form_data.weather %}selected{% endif %}>Select Weather</option>
                                {% for opt in ['Sunny', 'Rainy', 'Snowy', 'Cloudy'] %}
                                <option value="{{ opt }}" {% if form_data.weather == opt %}selected{% endif %}>{{ opt }}</option>
                                {% endfor %}
                            </select>
                        </div>

                    </div>
                </div>

                <!-- Section 2: Pricing & Competition -->
                <div>
                    <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                        <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        Pricing & Elasticity
                    </h2>
                    <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        
                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Our Price ($)</label>
                            <input type="number" step="0.01" name="price" placeholder="e.g. 75.00" value="{{ form_data.price }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Competitor Price ($)</label>
                            <input type="number" step="0.01" name="competitor_pricing" placeholder="e.g. 80.00" value="{{ form_data.competitor_pricing }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Discount Rate (%)</label>
                            <input type="number" step="1" name="discount" placeholder="e.g. 10" value="{{ form_data.discount }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                        </div>

                    </div>
                </div>

                <!-- Section 3: Inventory, Orders & Market Conditions -->
                <div>
                    <h2 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                        <svg class="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                        Supply & Market Signals
                    </h2>
                    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        
                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Current Stock Level</label>
                            <input type="number" name="inventory_level" placeholder="e.g. 200" value="{{ form_data.inventory_level }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Replenishment Units</label>
                            <input type="number" name="units_ordered" placeholder="e.g. 150" value="{{ form_data.units_ordered }}" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Promotional Campaign</label>
                            <select name="promotion" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                                <option value="" disabled {% if not form_data.promotion and form_data.promotion != '0' %}selected{% endif %}>Select Promo</option>
                                <option value="1" {% if form_data.promotion|string == '1' %}selected{% endif %}>Active Promo</option>
                                <option value="0" {% if form_data.promotion|string == '0' %}selected{% endif %}>Standard</option>
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-300 mb-1.5">Epidemic / Disruption</label>
                            <select name="epidemic" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition">
                                <option value="" disabled {% if not form_data.epidemic and form_data.epidemic != '0' %}selected{% endif %}>Select Status</option>
                                <option value="0" {% if form_data.epidemic|string == '0' %}selected{% endif %}>Normal State</option>
                                <option value="1" {% if form_data.epidemic|string == '1' %}selected{% endif %}>Crisis / Wave</option>
                            </select>
                        </div>

                    </div>
                </div>

                <!-- Action Buttons: Submit & Reset -->
                <div class="flex flex-col sm:flex-row gap-3 pt-2">
                    <button type="submit" class="flex-1 py-3.5 px-6 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-sm tracking-wide shadow-lg shadow-blue-500/25 active:scale-[0.99] transition duration-150 flex items-center justify-center gap-2">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                        Compute Forecasted Demand
                    </button>
                    
                    <a href="/" class="sm:w-36 py-3.5 px-6 rounded-xl bg-slate-700/80 hover:bg-slate-700 border border-slate-600 text-slate-300 font-semibold text-sm tracking-wide text-center active:scale-[0.99] transition duration-150 flex items-center justify-center gap-2">
                        <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                        Reset
                    </a>
                </div>

            </form>

            <!-- Results Banner (Hidden until computation) -->
            {% if prediction is not none %}
            <div class="mt-8 pt-8 border-t border-slate-700/60">
                <div class="bg-gradient-to-br from-blue-900/40 via-indigo-900/20 to-slate-900/80 border border-blue-500/30 rounded-2xl p-6 text-center shadow-inner relative overflow-hidden">
                    <div class="absolute -right-10 -bottom-10 w-40 h-40 bg-blue-500/10 rounded-full blur-2xl pointer-events-none"></div>
                    <span class="text-xs font-semibold text-blue-400 uppercase tracking-widest block mb-1">Inference Complete</span>
                    <div class="text-4xl sm:text-5xl font-black text-white tracking-tight my-2">
                        {{ prediction }} <span class="text-xl sm:text-2xl font-semibold text-blue-300">Units</span>
                    </div>
                    <p class="text-xs text-slate-400 mt-2">
                        Predicted daily consumer purchase volume based on selected pricing, promotion, and seasonal indicators.
                    </p>
                </div>
            </div>
            {% endif %}

        </div>

        <!-- Footer -->
        <footer class="mt-8 text-center text-xs text-slate-500">
            Retail Demand Intelligence Engine &bull; End-to-End Machine Learning System
        </footer>

    </div>
</body>
</html>
"""

# Default initial state: completely empty / nil values
NIL_FORM = {
    'category': '',
    'region': '',
    'seasonality': '',
    'weather': '',
    'price': '',
    'competitor_pricing': '',
    'discount': '',
    'inventory_level': '',
    'units_ordered': '',
    'promotion': '',
    'epidemic': ''
}

def prepare_features(data_dict):
    """Transforms raw dictionary inputs into the exact 25 model feature columns."""
    price = float(data_dict['price'])
    discount = float(data_dict['discount'])
    competitor_price = float(data_dict['competitor_pricing'])
    promotion = int(data_dict['promotion'])
    epidemic = int(data_dict['epidemic'])
    inv_level = float(data_dict['inventory_level'])
    units_ordered = float(data_dict['units_ordered'])
    
    category = data_dict['category']
    region = data_dict['region']
    weather = data_dict['weather']
    season = data_dict['seasonality']
    
    # Feature transformations
    units_ordered_log = np.log1p(max(0, units_ordered))
    inventory_level_sqrt = np.sqrt(max(0, inv_level))
    price_diff = competitor_price - price
    discount_amount = price * (discount / 100.0)
    price_ratio = price / (competitor_price + 1e-5)
    seasonality_encoded = SEASONALITY_MAP.get(season, 0)
    
    # Calendar approximations for incoming query
    month = 6
    day_of_week = 2
    quarter = 2
    is_weekend = 0

    features = {
        'Price': price,
        'Discount': discount,
        'Promotion': promotion,
        'Competitor Pricing': competitor_price,
        'Epidemic': epidemic,
        'Units_Ordered_log': units_ordered_log,
        'Inventory_Level_sqrt': inventory_level_sqrt,
        'Month': month,
        'DayOfWeek': day_of_week,
        'Quarter': quarter,
        'Is_Weekend': is_weekend,
        'Price_Diff': price_diff,
        'Discount_Amount': discount_amount,
        'Price_Ratio': price_ratio,
        'Seasonality_Encoded': seasonality_encoded,
        'Category_Electronics': 1 if category == 'Electronics' else 0,
        'Category_Furniture': 1 if category == 'Furniture' else 0,
        'Category_Groceries': 1 if category == 'Groceries' else 0,
        'Category_Toys': 1 if category == 'Toys' else 0,
        'Region_North': 1 if region == 'North' else 0,
        'Region_South': 1 if region == 'South' else 0,
        'Region_West': 1 if region == 'West' else 0,
        'Weather Condition_Rainy': 1 if weather == 'Rainy' else 0,
        'Weather Condition_Snowy': 1 if weather == 'Snowy' else 0,
        'Weather Condition_Sunny': 1 if weather == 'Sunny' else 0
    }
    return pd.DataFrame([features])

@app.route('/', methods=['GET'])
def home():
    # Fresh load: no predictions, empty form fields
    return render_template_string(HTML_TEMPLATE, prediction=None, form_data=NIL_FORM)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form
    feature_df = prepare_features(input_data)
    predicted_val = pipeline.predict(feature_df)[0]
    final_output = max(0, int(round(predicted_val)))
    # Retain the user's selected values alongside the computed prediction
    return render_template_string(HTML_TEMPLATE, prediction=final_output, form_data=input_data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json()
    feature_df = prepare_features(payload)
    pred = pipeline.predict(feature_df)[0]
    return jsonify({
        'status': 'success',
        'predicted_demand_units': max(0, int(round(pred)))
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
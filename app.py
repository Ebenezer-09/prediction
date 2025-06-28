import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration du dashboard
st.set_page_config(
    page_title="Dashboard Aurifère", 
    layout="wide", 
    page_icon="🥇",
    initial_sidebar_state="expanded"
)

# Style CSS (unchanged)
st.markdown("""
<style>
    :root {
        --primary: #1890ff;
        --success: #52c41a;
        --warning: #faad14;
        --error: #f5222d;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary);
    }
    .prediction-high {
        background: #f6ffed;
        border-left: 4px solid var(--success);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    }
    .prediction-medium {
        background: #fffbe6;
        border-left: 4px solid var(--warning);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    }
    .prediction-low {
        background: #fff2f0;
        border-left: 4px solid var(--error);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    }
    .model-btn {
        width: 100%;
        margin: 5px 0;
        padding: 10px;
        border-radius: 5px;
        text-align: left;
        transition: all 0.3s;
    }
    .model-btn:hover {
        background-color: #e6f7ff !important;
    }
    .active-model {
        background-color: var(--primary) !important;
        color: white !important;
    }
    .feature-slider {
        padding: 5px 0;
    }
    .stSlider > div {
        padding: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Chemin du dataset
DATASET_PATH = "donnees_minieres_5000_lignes.csv"

@st.cache_data
def load_and_prepare_data():
    try:
        data = pd.read_csv(DATASET_PATH)
        
        # Vérification des colonnes requises
        required_cols = ['profondeur', 'ph', 'conductivite', 'humidite', 'distance_faille', 'type_roche', 'teneur_or']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            st.error(f"Colonnes manquantes: {', '.join(missing)}")
            return None
        
        # Nettoyage des données
        data = data.dropna()
        
        # Préparation des données
        X = data[required_cols[:-1]].copy()  # Create a copy to avoid SettingWithCopyWarning
        y = data['teneur_or']
        
        # Encodage
        le = LabelEncoder()
        X.loc[:, 'type_roche'] = le.fit_transform(X['type_roche'])  # Use .loc for assignment
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train,
            'X_test_scaled': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'label_encoder': le,
            'features': required_cols[:-1],
            'target': 'teneur_or',
            'raw_data': data,
            'feature_ranges': {
                col: {
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'mean': float(data[col].mean())
                } for col in required_cols[:-1] if col != 'type_roche'
            }
        }
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        return None

# Initialisation des modèles
if 'models' not in st.session_state:
    st.session_state.models = {}

if 'data' not in st.session_state:
    data_dict = load_and_prepare_data()
    if data_dict:
        # Entraînement des modèles
        lr = LinearRegression()
        lr.fit(data_dict['X_train'], data_dict['y_train'])
        y_pred_lr = lr.predict(data_dict['X_test'])
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(data_dict['X_train'], data_dict['y_train'])
        y_pred_rf = rf.predict(data_dict['X_test'])
        
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        mlp.fit(data_dict['X_train_scaled'], data_dict['y_train'])
        y_pred_mlp = mlp.predict(data_dict['X_test_scaled'])
        
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        xgb.fit(data_dict['X_train_scaled'], data_dict['y_train'])
        y_pred_xgb = xgb.predict(data_dict['X_test_scaled'])
        
        # Store predictions in session state
        st.session_state.models = {
            'Régression Linéaire': {
                'model': lr,
                'predictions': y_pred_lr,  # Add predictions
                'metrics': {
                    'MAE': mean_absolute_error(data_dict['y_test'], y_pred_lr),
                    'RMSE': np.sqrt(mean_squared_error(data_dict['y_test'], y_pred_lr)),
                    'R2': r2_score(data_dict['y_test'], y_pred_lr)
                },
                'needs_scaling': False,
                'color': '#1890ff'
            },
            'Random Forest': {
                'model': rf,
                'predictions': y_pred_rf,  # Add predictions
                'metrics': {
                    'MAE': mean_absolute_error(data_dict['y_test'], y_pred_rf),
                    'RMSE': np.sqrt(mean_squared_error(data_dict['y_test'], y_pred_rf)),
                    'R2': r2_score(data_dict['y_test'], y_pred_rf)
                },
                'needs_scaling': False,
                'color': '#52c41a'
            },
            'MLP': {
                'model': mlp,
                'predictions': y_pred_mlp,  # Add predictions
                'metrics': {
                    'MAE': mean_absolute_error(data_dict['y_test'], y_pred_mlp),
                    'RMSE': np.sqrt(mean_squared_error(data_dict['y_test'], y_pred_mlp)),
                    'R2': r2_score(data_dict['y_test'], y_pred_mlp)
                },
                'needs_scaling': True,
                'color': '#722ed1'
            },
            'XGBoost': {
                'model': xgb,
                'predictions': y_pred_xgb,  # Add predictions
                'metrics': {
                    'MAE': mean_absolute_error(data_dict['y_test'], y_pred_xgb),
                    'RMSE': np.sqrt(mean_squared_error(data_dict['y_test'], y_pred_xgb)),
                    'R2': r2_score(data_dict['y_test'], y_pred_xgb)
                },
                'needs_scaling': True,
                'color': '#fa8c16'
            }
        }
        st.session_state.data = data_dict
        st.session_state.current_model = 'Random Forest'

# Sidebar - Navigation (unchanged)
with st.sidebar:
    st.markdown("## 🧠 Modèles Disponibles")
    
    for model_name, model_data in st.session_state.models.items():
        btn_class = "active-model" if model_name == st.session_state.current_model else ""
        if st.button(
            model_name, 
            key=f"btn_{model_name}",
            help=f"MAE: {model_data['metrics']['MAE']:.2f} | R²: {model_data['metrics']['R2']:.2f}"
        ):
            st.session_state.current_model = model_name
    
    st.markdown("---")
    st.markdown("### 🏆 Meilleurs Performances")
    
    def get_best_model(metric, reverse=False):
        models = st.session_state.models
        return sorted(models.items(), 
                     key=lambda x: x[1]['metrics'][metric], 
                     reverse=reverse)[0][0]
    
    st.markdown(f"**🥇 Meilleur MAE:** `{get_best_model('MAE')}`")
    st.markdown(f"**🥈 Meilleur RMSE:** `{get_best_model('RMSE')}`")
    st.markdown(f"**🥉 Meilleur R²:** `{get_best_model('R2', True)}`")
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques du Dataset")
    st.markdown(f"- **Échantillons:** {len(st.session_state.data['raw_data'])}")
    st.markdown(f"- **Variables:** {len(st.session_state.data['features'])}")
    st.markdown(f"- **Target:** `{st.session_state.data['target']}`")

# Main Content
if 'models' not in st.session_state:
    st.warning("Chargement des données et initialisation des modèles...")
else:
    st.title(f"🥇 Prédiction Aurifère - {st.session_state.current_model}")
    
    # Métriques du modèle sélectionné
    current_model = st.session_state.models[st.session_state.current_model]
    metrics = current_model['metrics']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📉 MAE (Erreur Absolue Moyenne)</h4>
            <h2>{metrics['MAE']:.3f} g/t</h2>
            <p>Plus la valeur est basse, mieux c'est</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 RMSE (Erreur Quadratique)</h4>
            <h2>{metrics['RMSE']:.3f}</h2>
            <p>Mesure des grandes erreurs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 R² Score</h4>
            <h2>{metrics['R2']:.3f}</h2>
            <p>1.0 = prédiction parfaite</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Formulaire de prédiction
    st.markdown("## 🔮 Prédiction Interactive")
    
    input_data = {}
    cols = st.columns(2)
    
    for i, feature in enumerate(st.session_state.data['features']):
        col = cols[i % 2]
        with col:
            if feature == 'type_roche':
                options = st.session_state.data['raw_data']['type_roche'].unique()
                input_data[feature] = st.selectbox(
                    f"**{feature}**", 
                    options,
                    help="Type de formation géologique"
                )
            else:
                ranges = st.session_state.data['feature_ranges'][feature]
                input_data[feature] = st.slider(
                    f"**{feature}**",
                    min_value=ranges['min'],
                    max_value=ranges['max'],
                    value=float(ranges['mean']),
                    step=float((ranges['max'] - ranges['min'])/100),
                    help=f"Plage typique: {ranges['min']:.1f} à {ranges['max']:.1f}"
                )
    
    if st.button("🚀 Lancer la Prédiction", type="primary", use_container_width=True):
        # Préparation des données
        input_df = pd.DataFrame([input_data])
        input_df['type_roche'] = st.session_state.data['label_encoder'].transform(input_df['type_roche'])
        
        # Ensure input_df has the same column order as training data
        input_df = input_df[st.session_state.data['features']]
        
        if current_model['needs_scaling']:
            input_scaled = st.session_state.data['scaler'].transform(input_df)
            prediction = current_model['model'].predict(input_scaled)
        else:
            # Convert input_df to numpy array to match training data format
            input_array = input_df.to_numpy()
            prediction = current_model['model'].predict(input_array)
        
        pred_value = prediction[0]
        
        # Interprétation de la prédiction
        if pred_value < 0:
            # Cas de prédiction négative
            st.markdown(f"""
            <div class="prediction-low">
                <h3>⚠️ Résultat de prédiction</h3>
                <h1 style="color: #f5222d;">{pred_value:.2f} g/t</h1>
                <p><strong>Interprétation :</strong> Probabilité extrêmement faible de présence aurifère</p>
                <p>Modèle utilisé: <strong>{st.session_state.current_model}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 Analyse détaillée", expanded=True):
                st.warning("""
                **Cette prédiction négative indique :**
                - Une très faible probabilité de trouver des quantités significatives d'or
                - Les caractéristiques géologiques sont défavorables
                - Recommandation : Éviter cette zone pour l'exploitation
                """)
                
                # Graphique de distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=st.session_state.data['raw_data']['teneur_or'],
                    nbinsx=50,
                    name="Distribution réelle",
                    marker_color='#1890ff'
                ))
                fig.add_vline(
                    x=0, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Seuil négatif"
                )
                fig.update_layout(
                    title="Distribution des teneurs en or dans le dataset",
                    xaxis_title="Teneur en or (g/t)",
                    yaxis_title="Fréquence"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        elif pred_value < 1.0:
            # Faible teneur
            proba = min(100, max(0, (pred_value / 1.0) * 70))  # 0-1g/t -> 0-70%
            
            st.markdown(f"""
            <div class="prediction-medium">
                <h3>📉 Résultat de prédiction</h3>
                <h1 style="color: #faad14;">{pred_value:.2f} g/t</h1>
                <p><strong>Probabilité de présence aurifère :</strong> {proba:.0f}%</p>
                <p>Modèle utilisé: <strong>{st.session_state.current_model}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 Analyse détaillée", expanded=True):
                st.info("""
                **Cette prédiction indique :**
                - Une faible teneur en or potentielle
                - Nécessite des analyses complémentaires
                - Rentabilité probablement faible pour une exploitation industrielle
                """)
                
                # Jauge de probabilité
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = proba,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilité de présence aurifère"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': proba
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            # Teneur significative
            proba = min(100, 70 + (pred_value / 10.0) * 30)  # 1g/t+ -> 70-100%
            
            st.markdown(f"""
            <div class="prediction-high">
                <h3>📈 Résultat de prédiction</h3>
                <h1 style="color: #52c41a;">{pred_value:.2f} g/t</h1>
                <p><strong>Probabilité de présence aurifère :</strong> {proba:.0f}%</p>
                <p>Modèle utilisé: <strong>{st.session_state.current_model}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 Analyse détaillée", expanded=True):
                st.success("""
                **Cette prédiction indique :**
                - Une teneur en or potentiellement exploitable
                - Zone prometteuse pour des analyses approfondies
                - Potentiel économique intéressant
                """)
                
                # Graphique de comparaison avec les seuils industriels
                fig = go.Figure()
                
                # Seuils industriels
                fig.add_vrect(
                    x0=0, x1=1, 
                    fillcolor="red", opacity=0.2,
                    annotation_text="Non rentable", 
                    annotation_position="top left"
                )
                fig.add_vrect(
                    x0=1, x1=3, 
                    fillcolor="orange", opacity=0.2,
                    annotation_text="Marginal", 
                    annotation_position="top left"
                )
                fig.add_vrect(
                    x0=3, x1=10, 
                    fillcolor="green", opacity=0.2,
                    annotation_text="Rentable", 
                    annotation_position="top left"
                )
                
                # Position de la prédiction
                fig.add_vline(
                    x=pred_value, 
                    line_dash="dash", 
                    line_color="black",
                    annotation_text="Notre prédiction"
                )
                
                fig.update_layout(
                    title="Interprétation industrielle de la teneur en or",
                    xaxis_title="Teneur en or (g/t)",
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=200
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Visualisation des performances
    st.markdown("## 📊 Comparaison des Modèles")
    
    tab1, tab2 = st.tabs(["📉 Métriques", "📈 Prédictions vs Réelles"])
    
    with tab1:
        fig = make_subplots(rows=1, cols=3, subplot_titles=("MAE", "RMSE", "R² Score"))
        
        for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
            fig.add_trace(
                go.Bar(
                    x=list(st.session_state.models.keys()),
                    y=[m['metrics'][metric] for m in st.session_state.models.values()],
                    name=metric,
                    marker_color=[m['color'] for m in st.session_state.models.values()],
                    text=[f"{m['metrics'][metric]:.3f}" for m in st.session_state.models.values()],
                    textposition='auto'
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            height=400, 
            showlegend=False,
            margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        
        y_test = st.session_state.data['y_test']
        max_val = max(y_test.max(), 10)  # Limite à 10g/t pour une meilleure visualisation
        
        for name, model_data in st.session_state.models.items():
            if name == st.session_state.current_model:
                # Mettre en évidence le modèle actuel
                fig.add_trace(
                    go.Scatter(
                        x=y_test,
                        y=model_data['predictions'],
                        mode='markers',
                        name=name,
                        marker=dict(
                            color=model_data['color'],
                            size=10,
                            line=dict(width=1, color='DarkSlateGrey')
                        )
                    )
                )
            else:
                # Modèles moins visibles
                fig.add_trace(
                    go.Scatter(
                        x=y_test,
                        y=model_data['predictions'],
                        mode='markers',
                        name=name,
                        marker=dict(
                            color=model_data['color'],
                            size=6,
                            opacity=0.4
                        )
                    )
                )
        
        # Ligne de référence
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Prédiction parfaite',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(
            xaxis_title='Valeurs Réelles (g/t)',
            yaxis_title='Prédictions (g/t)',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer (unchanged)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Dashboard développé avec ❤️ par [Votre Nom] - © 2023</p>
    <p>Technologies: Streamlit • Scikit-learn • XGBoost • Plotly</p>
</div>
""", unsafe_allow_html=True)
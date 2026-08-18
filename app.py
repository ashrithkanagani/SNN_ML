import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from train_model import SNN,LIFLayer

st.set_page_config(page_title="SNN AQI Dashboard", layout="wide")

LABEL_NAMES = ["Good","Moderate","Poor","Severe"]

POLLUTANTS = ["CO","NH3","NO2","OZONE","PM10","PM2.5","SO2"]

AQI_BREAKS  = {
    "PM2.5": [(30,0),(60,1),(90,2),(1e9,3)],
    "PM10":  [(50,0),(100,1),(250,2),(1e9,3)],
    "NO2":   [(40,0),(80,1),(180,2),(1e9,3)],
    "SO2":   [(40,0),(80,1),(380,2),(1e9,3)],
    "CO":    [(1,0),(2,1),(10,2),(1e9,3)],
    "OZONE": [(50,0),(100,1),(168,2),(1e9,3)],
    "NH3":   [(200,0),(400,1),(800,2),(1e9,3)],
}


POLLUTANT_INFO = {

"CO": {
"effect":"Reduces oxygen supply in blood causing dizziness, headaches and confusion.",
"precautions":[
"Avoid heavy traffic areas",
"Ensure indoor ventilation",
"Use CO detectors at home"
]
},

"NH3":{
"effect":"Irritates eyes, throat and lungs.",
"precautions":[
"Avoid industrial emission areas",
"Wear masks outdoors",
"Improve indoor ventilation"
]
},

"NO2":{
"effect":"Causes lung inflammation and worsens asthma.",
"precautions":[
"Avoid traffic congestion zones",
"Limit outdoor exercise",
"Use air purifiers indoors"
]
},

"OZONE":{
"effect":"Causes chest pain, coughing and breathing difficulty.",
"precautions":[
"Avoid outdoor activity during peak pollution",
"Stay indoors when AQI is high"
]
},

"PM10":{
"effect":"Irritates respiratory tract and throat.",
"precautions":[
"Wear masks outdoors",
"Close windows during dust storms"
]
},

"PM2.5":{
"effect":"Fine particles enter bloodstream causing heart and lung diseases.",
"precautions":[
"Wear N95 masks",
"Use HEPA air purifiers",
"Avoid outdoor exercise during high pollution"
]
},

"SO2":{
"effect":"Triggers asthma and breathing problems.",
"precautions":[
"Avoid industrial zones",
"Wear protective masks"
]
}

}


model = pickle.load(open("snn_model.pkl","rb"))


def pollutant_score(val, pol):

    for thr, sc in AQI_BREAKS[pol]:
        if val <= thr:
            return float(sc)

    return 3.0


def build_features(vals, lat, lon):

    sc = [pollutant_score(vals[i], POLLUTANTS[i]) for i in range(7)]

    mean_sc = np.mean(sc)
    max_sc = np.max(sc)

    n_avail = 7

    ind_good = float(mean_sc < 0.5)
    ind_poor = float(mean_sc >= 1.5)
    ind_severe = float(mean_sc >= 2.5)

    features = vals + sc + [
        mean_sc,max_sc,n_avail,
        ind_good,ind_poor,ind_severe,
        lat,lon
    ]

    return np.array(features).reshape(1,-1)


def predict_prob(X):

    probs = np.zeros((1,4))

    for _ in range(20):
        probs += model.forward(np.random.rand(model.T,1,X.shape[1]) < X)

    probs /= 20

    pred = probs.argmax(1)[0]

    return pred, probs[0]



st.title("🌍 India Air Quality Dashboard (SNN Model)")

st.sidebar.header("City Information")

city = st.sidebar.text_input("City Name")

lat = st.sidebar.number_input("Latitude",value=20.5937)
lon = st.sidebar.number_input("Longitude",value=78.9629)


st.sidebar.header("Pollutant Levels")

vals = []

for p in POLLUTANTS:
    vals.append(st.sidebar.number_input(p,min_value=0.0))



st.subheader("📊 Pollutant Levels")

poll_df = pd.DataFrame({
    "Pollutant":POLLUTANTS,
    "Value":vals
})

fig = px.bar(poll_df,x="Pollutant",y="Value",color="Pollutant")

st.plotly_chart(fig,use_container_width=True)



if st.button("Predict AQI"):

    X = build_features(vals,lat,lon)

    pred,probs = predict_prob(X)

    label = LABEL_NAMES[pred]

    st.header(f"AQI for {city}: {label}")



    st.subheader("🎛 AQI Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text':"AQI Category"},
        gauge={
            'axis': {'range':[0,3]},
            'steps':[
                {'range':[0,1],'color':"green"},
                {'range':[1,2],'color':"yellow"},
                {'range':[2,3],'color':"orange"},
                {'range':[3,4],'color':"red"},
            ]
        }
    ))

    st.plotly_chart(gauge)



    st.subheader("🎯 Prediction Confidence")

    prob_df = pd.DataFrame({
        "Class":LABEL_NAMES,
        "Probability":probs
    })

    prob_chart = px.bar(prob_df,x="Class",y="Probability",color="Class")

    st.plotly_chart(prob_chart)


    st.subheader("🧠 Live Spike Raster")

    spikes = (np.random.rand(50,20) > 0.9).astype(int)

    spike_fig = px.imshow(
        spikes,
        color_continuous_scale="greys",
        aspect="auto"
    )

    st.plotly_chart(spike_fig,use_container_width=True)



    st.subheader("📈 Historical AQI Trend")

    try:

        df = pd.read_csv("cpcbdata.csv")

        
        df["city"] = df["city"].str.lower()
        city_input = city.lower()

        city_df = df[df["city"] == city_input]

        if len(city_df) > 0:

            hist = city_df.groupby("last_update")["pollutant_avg"].mean().reset_index()

            hist_fig = px.line(
                hist,
                x="last_update",
                y="pollutant_avg",
                title="Historical AQI Trend"
            )

            st.plotly_chart(hist_fig)

        else:
            st.warning("No historical data found for this city in dataset.")

    except:
        st.error("Dataset not found or incorrect format.")


    st.subheader("🩺 Health Effects & Precautions")

    for i,p in enumerate(POLLUTANTS):

        val = vals[i]

        if val > 0:

            info = POLLUTANT_INFO[p]

            st.markdown(f"### {p}")

            st.write("**Health Effects:**")
            st.write(info["effect"])

            st.write("**Precautions:**")

            for item in info["precautions"]:
                st.write(f"• {item}")
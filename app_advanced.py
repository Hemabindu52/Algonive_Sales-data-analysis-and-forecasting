import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Airbnb Analytics Dashboard", layout="wide")
st.title("🏠 Airbnb NYC Analytics Dashboard")

file=st.sidebar.file_uploader("Upload AB_NYC_2019.csv",type="csv")
if file:
    df=pd.read_csv(file)
    st.sidebar.header("Filters")
    ng=st.sidebar.multiselect("Neighbourhood Group",sorted(df["neighbourhood_group"].dropna().unique()),default=sorted(df["neighbourhood_group"].dropna().unique()))
    rt=st.sidebar.multiselect("Room Type",sorted(df["room_type"].dropna().unique()),default=sorted(df["room_type"].dropna().unique()))
    pr=st.sidebar.slider("Price Range",0,int(df.price.max()),(0,min(1000,int(df.price.max()))))
    dff=df[df.neighbourhood_group.isin(ng)&df.room_type.isin(rt)&df.price.between(pr[0],pr[1])]

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Listings",len(dff))
    c2.metric("Average Price",f"${dff.price.mean():.2f}")
    c3.metric("Avg Reviews",f"{dff.number_of_reviews.mean():.1f}")
    c4.metric("Availability",f"{dff.availability_365.mean():.0f}")

    st.subheader("Filtered Data")
    st.dataframe(dff.head(100))

    st.plotly_chart(px.histogram(dff,x="price",nbins=50,title="Price Distribution"),use_container_width=True)
    st.plotly_chart(px.bar(dff.groupby("room_type",as_index=False).size(),x="room_type",y="size",title="Room Types"),use_container_width=True)
    st.plotly_chart(px.scatter(dff,x="number_of_reviews",y="price",color="room_type",
                               hover_data=["neighbourhood_group"],title="Reviews vs Price"),use_container_width=True)
    st.plotly_chart(px.box(dff,x="neighbourhood_group",y="price",color="neighbourhood_group",
                           title="Price by Neighbourhood"),use_container_width=True)
    st.plotly_chart(px.scatter_mapbox(dff,lat="latitude",lon="longitude",color="price",
                        hover_name="name",zoom=9,height=500,
                        mapbox_style="open-street-map",
                        title="Listing Locations"),use_container_width=True)

    st.header("Price Prediction")
    try:
        model=joblib.load("airbnb_model.pkl")
        room=st.selectbox("Encoded Room Type",[0,1,2,3])
        group=st.number_input("Encoded Neighbourhood Group",0,10,0)
        mn=st.number_input("Minimum Nights",1,365,2)
        rev=st.number_input("Number of Reviews",0,10000,10)
        rpm=st.number_input("Reviews per Month",0.0,50.0,1.0)
        avail=st.slider("Availability",0,365,100)
        if st.button("Predict"):
            X=pd.DataFrame([[room,group,mn,rev,rpm,avail]],columns=["room_type","neighbourhood_group","minimum_nights","number_of_reviews","reviews_per_month","availability_365"])
            st.success(f"Predicted Price: ${model.predict(X)[0]:.2f}")
    except:
        st.warning("airbnb_model.pkl not found.")
else:
    st.info("Upload the Airbnb dataset to start.")

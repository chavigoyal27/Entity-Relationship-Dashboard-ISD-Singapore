import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from pyvis.network import Network
import networkx as nx
from pyvis.network import Network
from transformers import pipeline
from textblob import TextBlob
import pycountry 
from sklearn.ensemble import IsolationForest
import nltk
from nltk.corpus import stopwords
import spacy
import plotly.graph_objects as go
from datetime import datetime
from transformers import pipeline
import re
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI  # Correct import
from langchain.chains import LLMChain
from tqdm import tqdm
import json
import csv
from openai import OpenAI
import os
import openai
import tempfile


    
@st.cache_data

def load_data(file=None):
    if file is None:
        file_path = "smu_final.csv"   # Default file
        # Adjusting the names to include the 'id' column
        data = pd.read_csv(file_path, header=None, names=["id", "entity1", "relationship", "entity2"])
    else:
        data = pd.read_csv(file, header=None, names=["id", "entity1", "relationship", "entity2"])
    
    # Handle missing values
    data = data.fillna("Unknown")
    
    # Show the columns and a preview to verify structure
    return data

# Load environment variables from .env file
load_dotenv()

# Set up Azure OpenAI credentials from environment variables
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_APIKEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["OPENAI_API_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Initialize Azure OpenAI model using LangChain
llm = AzureChatOpenAI(
    deployment_name=os.environ["OPENAI_API_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    temperature=0.2,
    model=os.getenv("AZURE_OPENAI_MODEL_NAME")  # Automatically use the model from .env
)

# for the the AI analysis section
summary_prompt = PromptTemplate(
    input_variables=["relevant_text"],
    template="""
    ### Dataset Context & Disclaimer
    Please note that the following analysis is based on a **limited dataset** with inherent **limitations**, such as:
    - **Data Quality**: The dataset may contain gaps, outdated information, or inaccuracies.
    - **Data Scope**: The dataset may not represent the entire spectrum of risk factors or may be biased toward specific countries or topics.
    - **Data Representativeness**: Insights are derived from the data provided, which may not fully capture the complexities of the geopolitical or economic landscape.

    ### Task: Analyze and Compare Risk Countries

    You will receive a list of countries and their associated **risk scores**, which are calculated based on various factors such as frequency of mentions and sentiment analysis. Your task is to:

    1. **Summarize** the risk insights for each country based on the dataset.
    2. Provide a **reasoning** for the risk scores, considering factors like the frequency of mentions, sentiment, and geopolitical/economic contexts.
    3. **Cross-reference** the findings with **external research**, academic sources, or relevant news to either **support** or **refute** the insights provided by the dataset.
        - If possible, provide **links to external sources** or research papers that confirm or challenge the analysis.

    ---

    ### Input Format (Example):  
    Risk Country: *[Country Name]*  
    Risk Score: *[Score Value]*  
    Mentions and Sentiment: *[Key Sentiment or Keywords]*

    ---

    ### Expected Output Format:  

    **1. Risk Summary for [Country Name]:**
    - **Risk Score**: [Score]
    - **Reasoning**: Provide insights into why the country has this risk score based on the dataset, such as the frequency of mentions, sentiment analysis, and any patterns observed.

    **2. External Research Cross-Reference:**
    - **Supporting Sources**:
        - Link 1: [Title of the Source] ([Link to Research Paper/News Article])
        - Link 2: [Title of the Source] ([Link to Research Paper/News Article])
    - **Refuting Sources**:
        - Link 1: [Title of the Source] ([Link to Research Paper/News Article])

    **3. Key Takeaways & Conclusion:**
    - Summarize the comparison between the dataset's analysis and external sources. Indicate whether the dataset findings align with broader research or if further investigation is needed.

    ---

    ### Example Output:

    **Risk Summary for Singapore:**
    - **Risk Score**: 7.8
    - **Reasoning**: Singapore has a high risk score primarily due to its frequent mentions related to economic tensions and geopolitical concerns. While the country has a strong economy, it is deeply connected to global trade, making it vulnerable to shifts in international relations, especially with China and the US.
    
    **External Research Cross-Reference:**
    - **Supporting Sources**:
        - [The Economist Intelligence Unit Report on Singapore](https://www.eiu.com/) - This source highlights Singapore’s strategic position in Southeast Asia and its vulnerability to geopolitical tensions, supporting the high risk score from the dataset.
    - **Refuting Sources**:
        - [World Bank Report on Singapore’s Stability](https://data.worldbank.org/country/singapore) - Contradicts the high-risk label by emphasizing the country’s political stability, legal frameworks, and low corruption, which suggests that it is one of the safest economies in the world.

    **Key Takeaways & Conclusion:**
    - The analysis suggests that while Singapore has a high risk score in the dataset, the external sources point to a more stable picture. The difference may stem from the focus on trade relations in the dataset, whereas broader research emphasizes political and economic stability. It’s important to note that while geopolitical factors play a significant role, Singapore’s strong economic institutions offer resilience against external shocks.

    ---

    ### **Further Recommendations:**
    Based on this analysis, we recommend that any decision-making be supplemented with real-time data and continuous monitoring of international relations. While this dataset offers valuable insights, external sources should be consulted regularly to ensure a more well-rounded understanding of the risk factors.

    Text to summarize: 
    {relevant_text}
    """
)


# Define the LLM chain for summarization
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

def generate_isd_insights(data):
    # Check if required columns are in the data
    if 'entity1' not in data.columns or 'relationship' not in data.columns:
        return "⚠️ 'entity1' or 'relationship' column is missing from the data."

    # Compute Risk Score using the function
    country_risk_scores = calculate_country_risk_scores(data)
    if country_risk_scores is None or country_risk_scores.empty:
        return "⚠️ No valid risk scores could be generated."

    # Get the Top 5 High-Risk Countries
    top_countries = country_risk_scores.nlargest(5, "Risk Score")["other_country"].tolist()
    
    # Extract text data related to high-risk countries
    relevant_text = ""
    for country in top_countries:
        country_data = data[(data['entity1'] == country) | (data['entity2'] == country)].head(5).to_string()
        relevant_text += f"\n[Risk Alert: {country}]\n" + country_data + "\n"
    
    # Ensure there is enough text to summarize
    if len(relevant_text.split()) < 50:
        return "⚠️ Not enough relevant data to generate insights at this time."

    # Generate summary using LangChain
    try:
        insights = summary_chain.run({"relevant_text": relevant_text})
        return insights
    except Exception as e:
        return f"⚠️ Error generating insights: {str(e)}"



# ✅ Function to check if an entity is a country
def is_country(name):
    return name in [country.name for country in pycountry.countries]

# ✅ Enhanced Risk Score Calculation for Countries
# List of valid countries
COUNTRIES = {
    "india", "singapore", "china", "south korea", "japan", "united states",
    "usa", "uk", "united kingdom", "australia", "germany", "france", "malaysia",
    "thailand", "philippines", "vietnam", "indonesia", "myanmar", "brunei",
    "laos", "cambodia", "bangladesh", "nepal", "pakistan"
}

def is_country(name):
    """Checks if an entity is a valid country."""
    return name in COUNTRIES

def calculate_country_risk_scores(data):
    """
    Calculates risk scores for countries interacting with Singapore based on sentiment analysis.
    """
    st.subheader("🌍 Risk Score Analysis for Countries with Respect to Singapore")

    # Convert all text to lowercase for consistency
    data = data.astype(str).apply(lambda x: x.str.lower())

    # Filter rows where Singapore is involved in either entity1 or entity2
    singapore_data = data[(data['entity1'] == "singapore") | (data['entity2'] == "singapore")]

    if singapore_data.empty:
        st.warning("⚠️ No valid data found for Singapore interactions.")
        return None

    # Extract the other country interacting with Singapore
    singapore_data["other_country"] = singapore_data.apply(
        lambda row: row["entity2"] if row["entity1"] == "singapore" else row["entity1"], axis=1
    )

    # Keep only valid country names
    singapore_data = singapore_data[singapore_data["other_country"].apply(is_country)]

    if singapore_data.empty:
        st.warning("⚠️ No valid countries found interacting with Singapore.")
        return None

    # Sentiment Analysis on 'relationship' column
    singapore_data["Sentiment"] = singapore_data["relationship"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
    )

    # Compute Frequency of Each Country's Interaction
    entity_counts = singapore_data["other_country"].value_counts().reset_index()
    entity_counts.columns = ["Country", "Frequency"]

    # Merge Frequency with original data
    singapore_data = singapore_data.merge(entity_counts, left_on="other_country", right_on="Country", how="left")

    # Compute Risk Score: Higher mentions + more negative sentiment = higher risk
    singapore_data["Risk Score"] = singapore_data["Frequency"] * (1 - singapore_data["Sentiment"])

    # Aggregate risk scores by country
    country_risk_scores = singapore_data.groupby("other_country").agg({"Risk Score": "sum"}).reset_index()

    # Get the Top 10 High-Risk Countries
    top_risk_countries = country_risk_scores.nlargest(10, "Risk Score")

    # Plot Risk Score Bar Chart
    fig = px.bar(
        top_risk_countries, x="other_country", y="Risk Score", color="Risk Score",
        title="🌍 Top 10 High-Risk Countries with Respect to Singapore",
        labels={"other_country": "Country Name", "Risk Score": "Total Risk Score"},
        color_continuous_scale="Reds"
    )
    
    st.plotly_chart(fig)

    return country_risk_scores


# ✅ Load NLP model for entity recognition
nlp = spacy.load("en_core_web_sm")

# ✅ Load stopwords and add custom filters
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
custom_exclusions = {"he", "we", "people", "man", "woman", "they", "it", "this", "these", "who", "whom"}

# ✅ Function to check if entity is meaningful
def is_valid_entity(name):
    return name.lower() not in stop_words and name.lower() not in custom_exclusions and len(name) > 2

# ✅ Function to classify entity as a person or company
def classify_entity(name):
    doc = nlp(name)
    for ent in doc.ents:
        if ent.label_ in ["ORG"]:  # Organization
            return "Company"
        elif ent.label_ in ["PERSON"]:  # Person
            return "Person"
    return None  # Ignore other entity types

# ✅ Function to detect anomalies in entity activity
def detect_anomalies(data):
    # Count entity occurrences
    entity_counts = data['entity1'].value_counts().reset_index()
    entity_counts.columns = ["Entity", "Count"]

    # ✅ Filter only meaningful entities
    entity_counts = entity_counts[entity_counts["Entity"].apply(is_valid_entity)]

    # Train Isolation Forest model
    model = IsolationForest(contamination=0.05)  # Flag top 5% as anomalies
    entity_counts["Anomaly Score"] = model.fit_predict(entity_counts[['Count']])

    # Filter for anomalous entities
    anomalies = entity_counts[entity_counts["Anomaly Score"] == -1]

    # Reduce entities to Top 10 most anomalous
    top_anomalies = anomalies.nlargest(20, "Count")

    # ✅ Categorize entities
    company_anomalies = []
    person_anomalies = []

    for index, row in top_anomalies.iterrows():
        entity_type = classify_entity(row["Entity"])
        if entity_type == "Company":
            company_anomalies.append(f"- *{row['Entity']}*: {row['Count']} mentions (Unusual Activity)")
        elif entity_type == "Person":
            person_anomalies.append(f"- *{row['Entity']}*: {row['Count']} mentions (Potential Risk)")

    # ✅ Visualization: Bar Chart for Anomalous Companies
    if company_anomalies:
        fig = px.bar(
            anomalies[anomalies["Entity"].apply(classify_entity) == "Company"],
            x="Entity", y="Count", color="Count",
            title="⚠️ Most Mentioned Companies & Organizations",
            labels={"Entity": "Company Name", "Count": "Number of Mentions"}
        )
        st.plotly_chart(fig)

    # ✅ Visualization: Bar Chart for Anomalous People
    if person_anomalies:
        fig = px.bar(
            anomalies[anomalies["Entity"].apply(classify_entity) == "Person"],
            x="Entity", y="Count", color="Count",
            title="⚠️ Most Mentioned Individuals",
            labels={"Entity": "Person Name", "Count": "Number of Mentions"}
        )
        st.plotly_chart(fig)


# ✅ Load NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# ✅ Function to Extract Dates from entity1 and entity2
def extract_dates(text):
    if pd.isna(text):  
        return None
    date_pattern = r'\b(19[0-9]{2}|20[0-9]{2})(?:[-/.](0[1-9]|1[0-2])(?:[-/.](0[1-9]|[12][0-9]|3[01]))?)?\b'
    match = re.search(date_pattern, str(text))
    if match:
        try:
            return datetime.strptime(match.group(), "%Y-%m-%d").date()
        except ValueError:
            return match.group(1)
    return None

# ✅ Function to Check if an Entity is a Meaningful Event
def is_meaningful_event(entity):
    doc = nlp(entity)
    for ent in doc.ents:
        if ent.label_ in ["DATE", "EVENT", "ORG", "PERSON"]:
            return True  # Only keep recognized events
    return False

#To extract the key events
event_extraction_prompt = """
### Task: Extract Key Events from the Dataset

You are provided with a dataset consisting of **entity relationships** in the format of "Entity 1", "Relationship", and "Entity 2". Your task is to extract key events from the provided data, considering the relationships and entities.

### Dataset Context:
The dataset contains a series of entity relationships. Each row consists of:
- **Entity1**: The first entity in the relationship.
- **Relationship**: The connection or interaction between the entities.
- **Entity2**: The second entity in the relationship.

Please extract key events by considering:
- The interactions between entities (what happened between them).
- Possible geopolitical, economic, or social events indicated by the relationships.
- Any additional context that provides more insight into the event.

The data will be provided as follows:
- **Entity 1**: {entity1}
- **Relationship**: {relationship}
- **Entity 2**: {entity2}

### Input Format:
Extract the key events based on the first 100 rows of the data:

{data}  # This will be the cleaned and formatted text from your dataset

### Expected Output:
Please summarize the key events, focusing on:
1. What major events or interactions are suggested by the relationships?
2. Any specific details about the events that should be highlighted (e.g., timing, impacts, or consequences).
3. Please ensure your response is concise and directly tied to the dataset provided.

---

### Example Output:

**Extracted Key Events:**

1. [Event 1 Description]
2. [Event 2 Description]
3. [Event 3 Description]

Provide as many details as possible about the key events in the dataset, and ensure that they are actionable insights.
"""

# Create a PromptTemplate object
prompt_template = PromptTemplate(
    input_variables=["data", "entity1", "relationship", "entity2"],  # Define inputs
    template=event_extraction_prompt
)

# Create the LLMChain to run with LangChain's LLM (Azure)
event_chain = LLMChain(llm=llm, prompt=prompt_template)

def extract_key_events(data):
    # Prepare the text data by joining relevant entity relationships
    text_data = data[['entity1', 'relationship', 'entity2']].astype(str).agg(' '.join, axis=1)
    text_data = ' '.join(text_data.head(100).tolist())  # Get the first 100 rows for summarization

    # Define variables that will be passed into the PromptTemplate
    entity1 = "Entity1 Example"  # You can dynamically select the relevant values from your data
    relationship = "Relationship Example"
    entity2 = "Entity2 Example"

    # Generate the prompt dynamically with provided variables
    prompt = prompt_template.format(data=text_data, entity1=entity1, relationship=relationship, entity2=entity2)

    try:
        # Run the LLMChain to get the response from the model
        summary = event_chain.run({
            "data": text_data,
            "entity1": entity1,
            "relationship": relationship,
            "entity2": entity2
        })
        return summary
    except Exception as e:
        return f"⚠️ Error extracting events: {str(e)}"


# ✅ Improved Timeline Function
def create_event_timeline(data):
    # Extract Dates
    data['date'] = data['entity1'].apply(extract_dates)
    data['date'].fillna(data['entity2'].apply(extract_dates), inplace=True)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date'])  

    if data.empty:
        st.warning("⚠️ No valid dates could be extracted.")
        return

    # # ✅ Extract Key Events using LLM
    key_events = extract_key_events(data)
    st.subheader("📢 Key Events")
    st.write(key_events)

    # ✅ Filter Only Meaningful Events
    data = data[data['entity1'].apply(is_meaningful_event)]
    data = data[data['entity2'].apply(is_meaningful_event)]

    # ✅ Create Timeline Visualization (No Right Column)
    fig = px.scatter(
        data, x="date", y="entity1",
        title="📅 Timeline of Key Entity Interactions",
        labels={"date": "Date", "entity1": "Entity"},
        height=600  
    )
    st.plotly_chart(fig) 

import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import re
import os

# Function to generate network graph
def generate_network_graph(data: pd.DataFrame, output: str = "display/interactive_graph.html"):

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges from data
    for _, row in data.iterrows():
        G.add_edge(row["entity1"], row["entity2"], label=row["relationship"])

    # Create Pyvis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)

    # Add nodes and edges to Pyvis network
    for node in G.nodes():
        net.add_node(node, label=node)

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2]["label"])

    # Save the interactive graph
    output_folder = "display"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "interactive_graph.html")
    net.save_graph(output_path)

    return output_path



def display_graph(image_path, caption, insight_text):
    with st.container():
        st.image(image_path, caption=caption, use_container_width=True)
        with st.expander("📌"):  # Expander is now placed directly below the image
            st.write(insight_text)




# ✅ Streamlit App Layout
def main():
    st.set_page_config(page_title="Entity Relationship Dashboard", layout="wide", initial_sidebar_state="expanded")
    
    # Apply custom theme (black/white)
    st.markdown("""
    <style>
    .css-1d391kg {background-color: #000; color: #fff;}
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg p {color: white;}
    .css-1d391kg .stButton {background-color: #fff; color: #000;}
    .css-1d391kg .stTextInput input, .css-1d391kg .stFileUploader input {color: white; background-color: #000;}
    .css-1d391kg .stSelectbox, .css-1d391kg .stRadio {color: white; background-color: #000;}
    .center-text {
            text-align: center;
            font-family: 'Mokoto', sans-serif; /* Sci-fi like font */
            font-size: 250px;
            color: #00ffcc; /* A galaxy-inspired color */
            text-shadow: 0px 0px 10px rgba(0, 255, 255, 0.8); /* Glowing effect */
        }
        .subtext {
            text-align: center;
            font-family: 'Mokoto', sans-serif;
            font-size: 200px;
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display the title and description
    st.markdown('<h1 class="center-text">INTELLEXIS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">Next-Generation Analytics Engine</p>', unsafe_allow_html=True)
    upload_option = st.selectbox("Select Your Dashboard", ["BIA_DashBoard", "Custom_DashBoard"])
    
    if upload_option == "BIA_DashBoard":
        data = load_data()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📂 Data View", "📊 Graphs & Analysis", "🔍 AI Insights", "🔗 Interactive Graphs"])
        with tab1:

            # View Raw Data
            with st.expander("📂 View Raw Data"):
                st.dataframe(data)

        with tab2:
            st.subheader("Exploratory Data Analysis")
            col1, col2 = st.columns(2)

            with col1:
                display_graph(
                    "graphs/sentiment_distribution.png", 
                    "Sentiment Distribution",
                    """
                    The sentiment analysis graph provides a breakdown of positive, neutral, and negative sentiments in news excerpts.

                    A high proportion of negative sentiment may indicate significant coverage of topics like terrorism, communal tensions, or threats to public safety.
                    Neutral sentiment likely corresponds to factual reporting, such as updates on security measures or legislative changes.
                    Positive sentiment might reflect progress in counter-terrorism, successful integration initiatives, or positive stories about societal resilience.
                    
                    **Actionable Recommendations:**
                    - Prioritize monitoring negative sentiment articles, as they could highlight emerging threats or public unease.
                    - Cross-reference sentiment trends with specific keywords like "radicalization," "cybersecurity threats," or "social harmony" to pinpoint high-risk areas for deeper investigation.
                    """
                )

                display_graph(
                    "graphs/polarity_subjectivity.png", 
                    "Polarity & Subjectivity",
                    """
                    - **Polarity distribution (blue)** skews towards neutrality (centered around 0).
                    - The majority of news excerpts have a neutral or slightly positive tone, indicating that news sources tend to present moderate reporting rather than extreme sentiment.
                    
                    **Actionable Insight for ISD:**  
                    This aligns with ISD's need to detect hidden bias or influence in seemingly neutral reports, as propaganda and misinformation often disguise themselves under a neutral tone.
                    """
                )

            with col2:
                display_graph(
                    "graphs/wordcloud.png", 
                    "WordCloud",
                    """
                    The word cloud reveals the most frequent terms in the dataset, offering a quick view of dominant themes in the media.

                    - If terms related to extremism (e.g., "radicalization," "terrorist") or societal division (e.g., "protest," "ethnic tension") are prominent, this indicates areas requiring closer scrutiny.
                    - Conversely, words like "cohesion," "security," or "resilience" suggest themes of societal stability and may provide opportunities to promote positive narratives.
                    
                    **Actionable Recommendations:**
                    - Use the word cloud as a starting point for keyword-based sentiment analysis to connect frequently discussed topics with sentiment trends.
                    - Identify whether any new or unexpected terms are gaining prominence, as these could indicate emerging narratives requiring ISD's attention.
                    """
                )

                display_graph(
                    "graphs/correlation_matrix.png", 
                    "Correlation Matrix",
                    """
                    - **Low correlation (0.31) between Polarity and Subjectivity.**
                    - This suggests that news excerpts with strong opinions (high subjectivity) do not necessarily have extreme sentiment (highly positive or negative polarity).

                    **Key Insight for ISD:**  
                    ISD should not assume that subjective articles are biased towards extreme sentiments—they may still be neutral in tone.  
                    This is relevant when analyzing propaganda, disinformation, or politically charged news, where subjective writing might appear neutral in sentiment.
                    """
                )

            col3, col4 = st.columns(2)

            with col3:
                display_graph(
                    "graphs/top_sources.png", 
                    "Top Sources",
                    """
                    The graph showing top sources indicates where most of the data originates. This can reveal:

                    - Sources with potential biases or agendas influencing public perception.
                    - Media outlets or platforms most actively shaping narratives around security issues.

                    **Actionable Recommendations:**
                    - Prioritize fact-checking and contextual analysis for data from sources with a history of sensationalism or bias.
                    - Collaborate with credible sources that produce balanced, high-quality reporting to shape public discourse and counter disinformation.
                    - Analyze whether certain platforms (e.g., social media) disproportionately influence public sentiment on sensitive issues, necessitating targeted digital interventions.
                    """
                )

            # Key Takeaways Section
            with st.expander("📂 Key Takeaways"):
                st.write(
                    """
                    ### Early Detection of Radicalization:
                    1. **Focus on monitoring keywords and sentiment trends** related to extremist ideologies.
                    - Use the word cloud and negative sentiment trends to identify vulnerable groups and regions.

                    ### Shaping Public Narratives:
                    2. **Amplify positive sentiment stories**, especially those that highlight social resilience, integration, or counter-extremism success stories.
                    - Collaborate with media outlets that are credible and widely trusted to mitigate the influence of sensationalist reporting.

                    ### Digital Media Monitoring:
                    3. **Assess whether social media platforms are driving public sentiment shifts**, especially if negative sentiments align with misinformation campaigns or extremist propaganda.
                    - Invest in tools to perform real-time monitoring of frequently discussed terms and sentiment spikes.

                    ### Policy Input:
                    4. **Use the findings to inform public outreach campaigns** or policy adjustments addressing specific themes in the media.
                    - Address gaps where positive security narratives could be better communicated to reduce public fear or misinformation.

                    5. **Neutral sentiment does not mean unbiased reporting**—covert influence can exist in neutral-toned but highly subjective news.
                    - Articles with high subjectivity and neutral polarity may indicate covert influence campaigns and should be further investigated.
                    - Propaganda and misinformation may not always have extreme sentiment polarity but might still push an agenda through subtle persuasion and subjective framing.
                    """
                )



        with tab3:
            
            insights = generate_isd_insights(data)
            st.write(insights)

            detect_anomalies(data)

            create_event_timeline(data) 

        with tab4:

            #Graph 1
            st.subheader("🔗 Entity-Relationship Connections")
            # Using iframe to display the interactive graph
            html_file_path = "display/interactive_graph_spaced_out_advanced.html"
            with open(html_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=550, scrolling=True)
            st.write("This comprehensive network graph maps all the extracted entities and their interrelationships, capturing a wide array of connections across organizations, individuals, and countries. By including all relationships in the dataset, it provides a broader view of how different entities intersect and influence each other. This holistic approach can uncover hidden patterns, such as alliances, power structures, or key figures linking multiple regions or sectors. It is useful for understanding complex global networks and the underlying connections shaping current global narratives.")
            # #Graph 2

            st.subheader("🔗 Org-Individual-Country Connections")
            # Using iframe to display the interactive graph
            html_file_path = "display/interactive_graph_with_relationships.html"
            with open(html_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=550, scrolling=True)
            st.write("This network graph visualizes the relationships between organizations, individuals, and countries as identified in news excerpts. The nodes represent entities, and the edges depict their connections—such as a country being associated with an individual or an organization. This graph highlights the influential players in the news and their interactions across borders, demonstrating how organizations and individuals drive global narratives, policy discussions, or collaborations. It can provide insights into geopolitical dynamics, corporate influence, and individual leadership across countries.")

            #Graph 2
            st.subheader("🔗 Country Mentions Frequency")
            
            # Using iframe to display the interactive graph
            html_file_path = "display/network_graph_clean.html"
            with open(html_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=480, scrolling=True)
            st.write("This graph focuses on the frequency of mentions of different countries in the news excerpts. By visualizing the volume of connections between countries, we can identify trends in international relations, such as which countries are most frequently discussed or involved in the media landscape. This could indicate political, economic, or social shifts, or spotlight areas of heightened global interest, such as diplomatic agreements, conflicts, or international events. The density of the connections shows countries that play a central role in current global affairs.")

            


    elif upload_option == "Custom_DashBoard":
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        if uploaded_file is not None:
            data = load_data(uploaded_file)

            tab1, tab2, tab3 = st.tabs(["📂 Data View", "⚠️ Highlights & Key Events", "🔗 Interactive Graphs"])
            with tab1:
                with st.expander("📂 View Raw Data"):
                    st.dataframe(data)
            
            with tab2:
                
                detect_anomalies(data)
                st.subheader("📅 Key Events")
                events = extract_key_events(data)
                st.write(events)

            with tab3:

                st.subheader("🔗 Entity-Relationship Connections")
                html_file_path = generate_network_graph(data)
                with open(html_file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=550, scrolling=True)
                

if __name__ == "__main__":
    main()

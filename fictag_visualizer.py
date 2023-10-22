from urllib.parse import unquote_plus, quote_plus
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import numpy as np

# fandoms
st.title('Fictag Visualizer')
st.write('This is a tool to visualize the data from the Fictag project.')

def str_to_id_list(s):
    if isinstance(s, float):
        return []
    if isinstance(s, int):
        return [s]
    if "+" in s:
        return set([int(x) for x in s.split("+") if x != ""])
    else:
        return [int(s)]

# load data
@st.cache_data
def load_data():
    df = pd.read_csv('fictag-scraper/data/processed/canonical_fandoms.csv')
    return df

@st.cache_data
def load_general_tags():
    df = pd.read_csv('fictag-scraper/data/processed/tags/_general_tags.csv.gz')
    return df

@st.cache_data
def load_fandom_tags(fandom_id):
    df = pd.read_csv(f'fictag-scraper/data/processed/tags/{fandom_id}/tags.csv.gz')
    return df

@st.cache_data
def load_works(fandom_id):
    work_path = Path(f'fictag-scraper/data/processed/works/{fandom_id}')
    # load all .csv files in the directory
    df = pd.concat([pd.read_csv(f) for f in work_path.glob('*.csv.gz')])
    df["general_tag_ids"] = df["general_tag_ids"].apply(str_to_id_list)
    df["fandom_tag_ids"] = df["fandom_tag_ids"].apply(str_to_id_list)
    df["computed_general_tag_ids"] = df["computed_general_tag_ids"].apply(str_to_id_list)
    df["computed_fandom_tag_ids"] = df["computed_fandom_tag_ids"].apply(str_to_id_list)
    return df

df = load_data()

# get fandom from url
url = st.experimental_get_query_params()
if 'fandom' in url:
    fandom = url['fandom'][0]
    # change from url encoding to normal
    fandom = unquote_plus(fandom)
    fandoms = sorted(df['tag'].unique())
    fandom_selected = st.selectbox('Select a fandom', fandoms, index=fandoms.index(fandom))
    if fandom_selected != fandom:
        fandom = fandom_selected
        # change to url encoding
        fandom_url = quote_plus(fandom)
        st.experimental_set_query_params(fandom=fandom_url)
else:
    fandoms = sorted(df['tag'].unique())
    fandom = st.selectbox('Select a fandom', fandoms)
    # change to url encoding
    fandom_url = quote_plus(fandom)
    st.experimental_set_query_params(fandom=fandom_url)

def remove_lead(x):
    if fandom in x:
        x = x.replace(f'({fandom})', "")
    if x.startswith(" + "):
        return x[3:]
    else:
        return x

row = df[df['tag'] == fandom]
count = row['count'].values[0]
st.write(f'There are **{count:,}** works in the **{fandom}** fandom.')

# show spinner while loading
with st.spinner('Loading works...'):
    works = load_works(row['hash'].values[0])

multiple_check = False

with st.expander('Filter by tags'):
    # load tags
    general_tags = load_general_tags()
    fandom_tags = load_fandom_tags(row['hash'].values[0])

    # add computed columns
    add_computed = st.checkbox('Add computed columns')

    # allow user to select multiple tags
    general_tags_selected = st.multiselect('Select general tags', general_tags['tag'].unique())
    fandom_tags_selected = st.multiselect('Select fandom tags', fandom_tags['tag'].unique())

    # allow selecting "any" or "all" mode
    mode = st.radio('Select mode', ['any', 'all'])

    # filter works
    general_tag_ids = []
    for tag in general_tags_selected:
        # the id is the index of the tag
        general_tag_ids.append(int(general_tags[general_tags['tag'] == tag].index[0]))

    fandom_tag_ids = []
    for tag in fandom_tags_selected:
        # the id is the index of the tag
        fandom_tag_ids.append(int(fandom_tags[fandom_tags['tag'] == tag].index[0]))

    if len(general_tag_ids) != 0 or len(fandom_tag_ids) != 0:
        if mode == 'any':
            # any of the tags
            if not add_computed:
                general_filter = works['general_tag_ids'].apply(lambda x: any([ tag in x for tag in general_tag_ids]))
                fandom_filter = works['fandom_tag_ids'].apply(lambda x: any([tag in x for tag in fandom_tag_ids]))
                works = works[general_filter | fandom_filter]
            else:
                general_filter_comp = works['computed_general_tag_ids'].apply(lambda x: any([ tag in x for tag in general_tag_ids]))
                fandom_filter_comp = works['computed_fandom_tag_ids'].apply(lambda x: any([tag in x for tag in fandom_tag_ids]))
                general_filter = works['general_tag_ids'].apply(lambda x: any([ tag in x for tag in general_tag_ids]))
                fandom_filter = works['fandom_tag_ids'].apply(lambda x: any([tag in x for tag in fandom_tag_ids]))
                general_filter = general_filter | general_filter_comp
                fandom_filter = fandom_filter | fandom_filter_comp
                works = works[general_filter | fandom_filter]
        elif mode == 'all':
            # all of the tags
            if not add_computed:
                general_filter = works['general_tag_ids'].apply(lambda x: all([tag in x for tag in general_tag_ids]))
                fandom_filter = works['fandom_tag_ids'].apply(lambda x: all([tag in x for tag in fandom_tag_ids]))
                works = works[general_filter & fandom_filter]
            else:
                general_filter_comp = works['computed_general_tag_ids'].apply(lambda x: all([tag in x for tag in general_tag_ids]))
                fandom_filter_comp = works['computed_fandom_tag_ids'].apply(lambda x: all([tag in x for tag in fandom_tag_ids]))
                general_filter = works['general_tag_ids'].apply(lambda x: all([tag in x for tag in general_tag_ids]))
                fandom_filter = works['fandom_tag_ids'].apply(lambda x: all([tag in x for tag in fandom_tag_ids]))
                general_filter = general_filter | general_filter_comp
                fandom_filter = fandom_filter | fandom_filter_comp
                works = works[general_filter & fandom_filter]
        multiple_check = st.checkbox('Show multiple tags')
        if multiple_check:
            # means multiple lines in the plot
            if add_computed:
                general_tags_l = []
                fandom_tags_l = []
                for _, row in works.iterrows():
                    # set union
                    general_tags_l.append(set(row['general_tag_ids']) | set(row['computed_general_tag_ids']))
                    fandom_tags_l.append(set(row['fandom_tag_ids']) | set(row['computed_fandom_tag_ids']))
                works['general_tag_ids'] = general_tags_l
                works['fandom_tag_ids'] = fandom_tags_l
            general_str = works['general_tag_ids'].apply(lambda x: " + ".join(sorted([general_tags.loc[tag]['tag'] for tag in x if tag in general_tag_ids])))
            fandom_str = works['fandom_tag_ids'].apply(lambda x: " + ".join(sorted([fandom_tags.loc[tag]['tag'] for tag in x if tag in fandom_tag_ids])))
            works['group'] = general_str + " + " + fandom_str
            works['group'] = works['group'].apply(remove_lead)

    st.write(f'There are **{len(works):,}** works that match the selected tags.')

# plot
with st.expander('Show histogram of word counts'):
    # remove outliers
    remove_outliers = st.checkbox('Remove outliers')
    if remove_outliers:
        quantile = st.slider('Upper and lower quantile', 0.0, 1.0, (0.01, 0.99), 0.01)
        works = works[(works['words'] > works['words'].quantile(quantile[0])) & (works['words'] < works['words'].quantile(quantile[1]))]
    fig = px.histogram(works, x="words", nbins=100, title=f'Word count for {fandom}')
    st.plotly_chart(fig, use_container_width=True)

# plot over time
with st.expander('Show works over time'):
    # select aggregation value (count, words, kudos, etc.)
    aggregation = st.selectbox('Select aggregation', ['count', 'words', 'kudos', 'comments', 'bookmarks', 'hits'])
    # select time period
    time_period = st.selectbox('Select time period', ['day', 'month', 'year'])
    

    # convert to datetime
    works['date'] = pd.to_datetime(works['date'])
    # remove all >= 2023 (probably errors)
    works = works[works['date'].dt.year < 2023]
    # remove all <= 2007 (probably errors)
    works = works[works['date'].dt.year > 2007]
    # add count column
    works['count'] = 1
    # group by time period
    if time_period == 'day':
        works['date'] = works['date'].dt.date
    elif time_period == 'month':
        # e.g. 2021-01, 2021-02, etc.
        works['date'] = works['date'].dt.strftime('%Y-%m')
    elif time_period == 'year':
        works['date'] = works['date'].dt.strftime('%Y')


    # group by date
    if multiple_check:
        st.write(works)
        works_g = works.groupby(['date', 'group']).agg({aggregation: 'sum'}).reset_index()
    else:
        works_g = works.groupby('date').agg({aggregation: 'sum'}).reset_index()

    # plot
    title = f'Number of {aggregation} per {time_period} for {fandom}'
    if len(general_tag_ids) != 0 or len(fandom_tag_ids) != 0:
        title += ' (filtered by {} tags)'.format(fandom_tags_selected+general_tags_selected)

    if multiple_check:
        fig = px.line(works_g, x="date", y=aggregation, title=title, line_group='group', color='group', markers=True)
    else:
        fig = px.line(works_g, x="date", y=aggregation, title=title)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Show tag co-occurrence"):
    selected_tag = st.selectbox('Select tag', fandom_tags['tag'].unique())
    selected_tag_id = fandom_tags[fandom_tags['tag'] == selected_tag].index[0]
    # get all works with this tag
    works_with_tag = works[works['fandom_tag_ids'].apply(lambda x: selected_tag_id in x)|works['computed_fandom_tag_ids'].apply(lambda x: selected_tag_id in x)]
    # get all other tags
    tag_occurrences = {}
    general_tag_occurrences = {}
    progress_bar = st.progress(0, text='Counting tag occurrences')
    total_len = len(works_with_tag)
    for j, row in works_with_tag.iterrows():
        for tag in row['fandom_tag_ids']:
            if tag != selected_tag_id:
                if tag not in tag_occurrences:
                    tag_occurrences[tag] = 0
                tag_occurrences[tag] += 1
        for tag in row['computed_fandom_tag_ids']:
            if tag != selected_tag_id:
                if tag not in tag_occurrences:
                    tag_occurrences[tag] = 0
                tag_occurrences[tag] += 1
        for tag in row['general_tag_ids']:
            if tag not in general_tag_occurrences:
                general_tag_occurrences[tag] = 0
            general_tag_occurrences[tag] += 1
        for tag in row['computed_general_tag_ids']:
            if tag not in general_tag_occurrences:
                general_tag_occurrences[tag] = 0
            general_tag_occurrences[tag] += 1
        progress_bar.progress(j / total_len, text=f'Counting tag occurrences ({j:,}/{total_len:,})')
    # divide by number of tag occurrences
    scale_by_idf = st.checkbox('Scale by IDF', value=True)
    if scale_by_idf:
        for tag in tag_occurrences:
            works_with_this_tag = works[works['fandom_tag_ids'].apply(lambda x: tag in x)|works['computed_fandom_tag_ids'].apply(lambda x: tag in x)]
            idf = np.log(len(works) / len(works_with_this_tag))
            tf = tag_occurrences[tag] / len(works_with_tag)
            tag_occurrences[tag] = tf * idf
        for tag in general_tag_occurrences:
            works_with_this_tag = works[works['general_tag_ids'].apply(lambda x: tag in x)|works['computed_general_tag_ids'].apply(lambda x: tag in x)]
            idf = np.log(len(works) / len(works_with_this_tag))
            tf = general_tag_occurrences[tag] / len(works_with_tag)
            general_tag_occurrences[tag] = tf * idf
    # create dataframe
    tag_occurrences = pd.DataFrame.from_dict(tag_occurrences, orient='index', columns=['value'])
    tag_occurrences = tag_occurrences.sort_values('value', ascending=False)
    general_tag_occurrences = pd.DataFrame.from_dict(general_tag_occurrences, orient='index', columns=['value'])
    general_tag_occurrences = general_tag_occurrences.sort_values('value', ascending=False)
    # add tag names
    tag_occurrences['tag'] = tag_occurrences.index.map(lambda x: fandom_tags.loc[x]['tag'])
    # use remove_lead function from above
    tag_occurrences['tag'] = tag_occurrences['tag'].apply(remove_lead)
    tag_occurrences['category'] = tag_occurrences.index.map(lambda x: fandom_tags.loc[x]['category'])
    general_tag_occurrences['tag'] = general_tag_occurrences.index.map(lambda x: general_tags.loc[x]['tag'])
    filter_out = st.multiselect('Filter out categories', tag_occurrences['category'].unique())
    if len(filter_out) != 0:
        tag_occurrences = tag_occurrences[~tag_occurrences['category'].isin(filter_out)]
    
    # plot
    show_top = st.slider('Show top', 1, 100, 10)
    fig = px.pie(tag_occurrences[:show_top], values='value', names='tag', title=f'Co-occurrence of {selected_tag} with fandom tags')
    st.plotly_chart(fig, use_container_width=True)
    fig = px.pie(general_tag_occurrences[:show_top], values='value', names='tag', title=f'Co-occurrence of {selected_tag} with general tags')
    st.plotly_chart(fig, use_container_width=True)
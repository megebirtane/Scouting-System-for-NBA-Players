import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="NBA Player Scouting Dashboard",
    page_icon="🏀",
    layout="wide"
)

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "AllStar_Dataset.csv")

@st.cache_data
def load_data():
    """Load and return the NBA player dataset."""
    df = pd.read_csv(CSV_PATH)
    return df

def save_player_to_csv(player_data):
    """Save a new player to the CSV file."""
    df = pd.read_csv(CSV_PATH)
    new_row = pd.DataFrame([player_data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    return True

def delete_player_from_csv(player_name):
    """Delete a player from the CSV file by name."""
    df = pd.read_csv(CSV_PATH)
    df = df[df['player_name'] != player_name]
    df.to_csv(CSV_PATH, index=False)
    return True

def update_player_in_csv(original_name, player_data):
    """Update a player's data in the CSV file."""
    df = pd.read_csv(CSV_PATH)
    # Find the index of the player to update
    idx = df[df['player_name'] == original_name].index
    if len(idx) > 0:
        for key, value in player_data.items():
            df.loc[idx[0], key] = value
        df.to_csv(CSV_PATH, index=False)
        return True
    return False

def calculate_metrics(df):
    """Calculate CEM and MVPCEM metrics for all players."""
    df = df.copy()

    # Normalize features for CEM calculation
    features_cem = ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'to', 'fg%', 'ft%']
    for col in features_cem:
        df[f'norm_{col}'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Calculate CEM
    df['CEM'] = (0.35 * df['norm_ppg'] + 0.20 * df['norm_rpg'] + 0.20 * df['norm_apg'] +
                 0.05 * df['norm_spg'] + 0.05 * df['norm_bpg'] - 0.10 * df['norm_to'] +
                 0.10 * df['norm_fg%'] + 0.05 * df['norm_ft%'])

    # Adjust CEM by games played
    df['Final_CEM'] = df['CEM'] * np.log(df['games_played'])

    # Normalize additional features for MVPCEM
    df['norm_3p%'] = (df['3p%'] - df['3p%'].min()) / (df['3p%'].max() - df['3p%'].min())
    df['norm_2p%'] = (df['2p%'] - df['2p%'].min()) / (df['2p%'].max() - df['2p%'].min())

    # Calculate MVPCEM
    df['PER_like'] = (0.35 * df['norm_ppg'] + 0.20 * df['norm_rpg'] + 0.20 * df['norm_apg'] +
                      0.05 * df['norm_spg'] + 0.05 * df['norm_bpg'] + 0.10 * df['norm_fg%'] +
                      0.07 * df['norm_3p%'] + 0.05 * df['norm_2p%'] + 0.05 * df['norm_ft%'] -
                      0.10 * df['norm_to'])

    df['MVPCEM'] = df['PER_like'] * np.log(df['games_played'])

    return df

def classify_players(df, metric='Final_CEM'):
    """Classify players into categories based on their metric scores."""
    df = df.copy()
    df_sorted = df.sort_values(by=metric, ascending=False)

    categories = ['MVP Caliber', 'All-NBA Caliber', 'All-Star Caliber']
    df_sorted['Category'] = pd.qcut(df_sorted[metric], 3, labels=categories[::-1])

    return df_sorted

def calculate_single_player_metrics(player_data, df):
    """Calculate CEM and MVPCEM for a single player compared to existing dataset."""
    # Combine the new player with existing data for normalization
    df_combined = pd.concat([df, pd.DataFrame([player_data])], ignore_index=True)

    # Normalize features for CEM calculation
    features_cem = ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'to', 'fg%', 'ft%']
    for col in features_cem:
        col_min = df_combined[col].min()
        col_max = df_combined[col].max()
        if col_max - col_min > 0:
            df_combined[f'norm_{col}'] = (df_combined[col] - col_min) / (col_max - col_min)
        else:
            df_combined[f'norm_{col}'] = 0

    # Normalize 3p% and 2p%
    for col in ['3p%', '2p%']:
        col_min = df_combined[col].min()
        col_max = df_combined[col].max()
        if col_max - col_min > 0:
            df_combined[f'norm_{col}'] = (df_combined[col] - col_min) / (col_max - col_min)
        else:
            df_combined[f'norm_{col}'] = 0

    # Get the last row (new player)
    new_player = df_combined.iloc[-1]

    # Calculate CEM
    cem = (0.35 * new_player['norm_ppg'] + 0.20 * new_player['norm_rpg'] + 0.20 * new_player['norm_apg'] +
           0.05 * new_player['norm_spg'] + 0.05 * new_player['norm_bpg'] - 0.10 * new_player['norm_to'] +
           0.10 * new_player['norm_fg%'] + 0.05 * new_player['norm_ft%'])

    final_cem = cem * np.log(new_player['games_played'])

    # Calculate MVPCEM
    per_like = (0.35 * new_player['norm_ppg'] + 0.20 * new_player['norm_rpg'] + 0.20 * new_player['norm_apg'] +
                0.05 * new_player['norm_spg'] + 0.05 * new_player['norm_bpg'] + 0.10 * new_player['norm_fg%'] +
                0.07 * new_player['norm_3p%'] + 0.05 * new_player['norm_2p%'] + 0.05 * new_player['norm_ft%'] -
                0.10 * new_player['norm_to'])

    mvpcem = per_like * np.log(new_player['games_played'])

    return final_cem, mvpcem

def create_rankings_chart(df, metric, title):
    """Create a horizontal bar chart for player rankings."""
    # Define colors for each category
    color_map = {
        'MVP Caliber': '#FFD700',      # Gold
        'All-NBA Caliber': '#C0C0C0',  # Silver
        'All-Star Caliber': '#CD7F32'  # Bronze
    }

    df_sorted = df.sort_values(by=metric, ascending=True)

    fig = go.Figure()

    for category in ['All-Star Caliber', 'All-NBA Caliber', 'MVP Caliber']:
        mask = df_sorted['Category'] == category
        fig.add_trace(go.Bar(
            y=df_sorted[mask]['player_name'],
            x=df_sorted[mask][metric],
            orientation='h',
            name=category,
            marker_color=color_map[category],
            text=df_sorted[mask][metric].round(2),
            textposition='outside'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=metric,
        yaxis_title="Player",
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='overlay'
    )

    return fig

# Main app
def main():
    st.title("NBA Player Scouting Dashboard")
    st.markdown("---")

    # Load data
    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Conference filter
    conference_options = ['All', 'EAST', 'WEST']
    selected_conference = st.sidebar.selectbox("Select Conference", conference_options)

    # Metric selector
    metric_options = {
        'Final CEM': 'Final_CEM',
        'MVPCEM': 'MVPCEM'
    }
    selected_metric_name = st.sidebar.selectbox("Select Metric", list(metric_options.keys()))
    selected_metric = metric_options[selected_metric_name]

    # Calculate metrics
    df_metrics = calculate_metrics(df)

    # Filter by conference
    if selected_conference != 'All':
        df_filtered = df_metrics[df_metrics['conference'] == selected_conference]
    else:
        df_filtered = df_metrics

    # Classify players
    df_classified = classify_players(df_filtered, selected_metric)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Player Rankings", "Add New Player", "Manage Players"])

    with tab1:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Players", len(df_classified))
        with col2:
            mvp_count = len(df_classified[df_classified['Category'] == 'MVP Caliber'])
            st.metric("MVP Caliber", mvp_count)
        with col3:
            top_player = df_classified.iloc[0]['player_name']
            st.metric("Top Player", top_player)

        st.markdown("---")

        # Player Rankings Bar Chart
        st.subheader(f"Player Rankings by {selected_metric_name}")

        chart_title = f"NBA Player Rankings - {selected_metric_name}"
        if selected_conference != 'All':
            chart_title += f" ({selected_conference} Conference)"

        fig = create_rankings_chart(df_classified, selected_metric, chart_title)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Top Players Summary Table
        st.subheader("Top Players Summary")

        display_cols = ['player_name', 'conference', 'ppg', 'rpg', 'apg', 'Category', selected_metric]
        df_display = df_classified[display_cols].copy()
        df_display.columns = ['Player', 'Conference', 'PPG', 'RPG', 'APG', 'Category', selected_metric_name]
        df_display[selected_metric_name] = df_display[selected_metric_name].round(2)

        st.dataframe(df_display, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Add New Player to Dataset")
        st.markdown("Enter all player statistics below to calculate their CEM rating and add them to the scouting database.")

        with st.form("add_player_form"):
            st.markdown("### Player Information")
            col1, col2 = st.columns(2)

            with col1:
                player_name = st.text_input("Player Name *", placeholder="e.g., John Smith")
            with col2:
                conference = st.selectbox("Conference *", options=["EAST", "WEST"])

            st.markdown("### Game Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                games_played = st.number_input("Games Played *", min_value=1, max_value=82, value=70, step=1)
                mp = st.number_input("Minutes Per Game *", min_value=1.0, max_value=48.0, value=30.0, step=0.1)
                ppg = st.number_input("Points Per Game (PPG) *", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
                rpg = st.number_input("Rebounds Per Game (RPG) *", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
                apg = st.number_input("Assists Per Game (APG) *", min_value=0.0, max_value=15.0, value=4.0, step=0.1)

            with col2:
                spg = st.number_input("Steals Per Game (SPG) *", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
                bpg = st.number_input("Blocks Per Game (BPG) *", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
                to = st.number_input("Turnovers Per Game (TO) *", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

            with col3:
                fg_pct = st.number_input("Field Goal % (FG%) *", min_value=0.0, max_value=1.0, value=0.45, step=0.001, format="%.3f")
                three_pct = st.number_input("3-Point % (3P%) *", min_value=0.0, max_value=1.0, value=0.35, step=0.001, format="%.3f")
                two_pct = st.number_input("2-Point % (2P%) *", min_value=0.0, max_value=1.0, value=0.50, step=0.001, format="%.3f")
                ft_pct = st.number_input("Free Throw % (FT%) *", min_value=0.0, max_value=1.0, value=0.80, step=0.001, format="%.3f")

            st.markdown("### Financial Information")
            salary = st.number_input("Salary ($) *", min_value=0, max_value=100000000, value=10000000, step=100000)

            st.markdown("---")
            submitted = st.form_submit_button("Calculate CEM & Add Player", use_container_width=True)

            if submitted:
                # Validate required fields
                if not player_name or player_name.strip() == "":
                    st.error("Please enter a player name.")
                elif player_name.strip() in df['player_name'].values:
                    st.error(f"A player named '{player_name.strip()}' already exists in the dataset.")
                else:
                    # Create player data dictionary
                    player_data = {
                        'player_name': player_name.strip(),
                        'conference': conference,
                        'games_played': int(games_played),
                        'mp': float(mp),
                        'ppg': float(ppg),
                        'rpg': float(rpg),
                        'apg': float(apg),
                        'spg': float(spg),
                        'bpg': float(bpg),
                        'to': float(to),
                        'fg%': float(fg_pct),
                        '3p%': float(three_pct),
                        '2p%': float(two_pct),
                        'ft%': float(ft_pct),
                        'Salary': int(salary)
                    }

                    # Calculate metrics for the new player
                    final_cem, mvpcem = calculate_single_player_metrics(player_data, df)

                    # Display calculated metrics
                    st.success(f"CEM Ratings calculated for {player_name}!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Final CEM Score", f"{final_cem:.2f}")
                    with col2:
                        st.metric("MVPCEM Score", f"{mvpcem:.2f}")

                    # Save to CSV
                    try:
                        save_player_to_csv(player_data)
                        st.success(f"Player '{player_name}' has been added to the dataset successfully!")
                        st.info("Please refresh the page or switch tabs to see the updated rankings.")
                        # Clear cache to reload data
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Error saving player: {str(e)}")

    with tab3:
        st.subheader("Manage Players")
        st.markdown("View, edit, and delete players from the dataset.")

        # Display all players in a table
        st.markdown("### Current Players in Dataset")

        # Create a display dataframe with key stats
        df_manage = df[['player_name', 'conference', 'ppg', 'rpg', 'apg', 'Salary']].copy()
        df_manage.columns = ['Player Name', 'Conference', 'PPG', 'RPG', 'APG', 'Salary']
        df_manage['Salary'] = df_manage['Salary'].apply(lambda x: f"${x:,}")

        st.dataframe(df_manage, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Create sub-tabs for Edit and Delete
        manage_tab1, manage_tab2 = st.tabs(["Edit Player", "Delete Player"])

        with manage_tab1:
            st.markdown("### Edit Player Stats")

            # Player selection for editing
            player_list = sorted(df['player_name'].tolist())
            edit_player = st.selectbox(
                "Select a player to edit",
                options=["-- Select a player --"] + player_list,
                key="edit_player_select"
            )

            if edit_player != "-- Select a player --":
                # Get current player data
                player_info = df[df['player_name'] == edit_player].iloc[0]

                with st.form("edit_player_form"):
                    st.markdown(f"**Editing:** {edit_player}")

                    st.markdown("#### Player Information")
                    col1, col2 = st.columns(2)

                    with col1:
                        new_player_name = st.text_input("Player Name *", value=player_info['player_name'])
                    with col2:
                        conf_index = 0 if player_info['conference'] == "EAST" else 1
                        new_conference = st.selectbox("Conference *", options=["EAST", "WEST"], index=conf_index, key="edit_conf")

                    st.markdown("#### Game Statistics")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        new_games_played = st.number_input("Games Played *", min_value=1, max_value=82, value=int(player_info['games_played']), step=1, key="edit_gp")
                        new_mp = st.number_input("Minutes Per Game *", min_value=1.0, max_value=48.0, value=float(player_info['mp']), step=0.1, key="edit_mp")
                        new_ppg = st.number_input("Points Per Game (PPG) *", min_value=0.0, max_value=50.0, value=float(player_info['ppg']), step=0.1, key="edit_ppg")
                        new_rpg = st.number_input("Rebounds Per Game (RPG) *", min_value=0.0, max_value=20.0, value=float(player_info['rpg']), step=0.1, key="edit_rpg")
                        new_apg = st.number_input("Assists Per Game (APG) *", min_value=0.0, max_value=15.0, value=float(player_info['apg']), step=0.1, key="edit_apg")

                    with col2:
                        new_spg = st.number_input("Steals Per Game (SPG) *", min_value=0.0, max_value=5.0, value=float(player_info['spg']), step=0.1, key="edit_spg")
                        new_bpg = st.number_input("Blocks Per Game (BPG) *", min_value=0.0, max_value=5.0, value=float(player_info['bpg']), step=0.1, key="edit_bpg")
                        new_to = st.number_input("Turnovers Per Game (TO) *", min_value=0.0, max_value=10.0, value=float(player_info['to']), step=0.1, key="edit_to")

                    with col3:
                        new_fg_pct = st.number_input("Field Goal % (FG%) *", min_value=0.0, max_value=1.0, value=float(player_info['fg%']), step=0.001, format="%.3f", key="edit_fg")
                        new_three_pct = st.number_input("3-Point % (3P%) *", min_value=0.0, max_value=1.0, value=float(player_info['3p%']), step=0.001, format="%.3f", key="edit_3p")
                        new_two_pct = st.number_input("2-Point % (2P%) *", min_value=0.0, max_value=1.0, value=float(player_info['2p%']), step=0.001, format="%.3f", key="edit_2p")
                        new_ft_pct = st.number_input("Free Throw % (FT%) *", min_value=0.0, max_value=1.0, value=float(player_info['ft%']), step=0.001, format="%.3f", key="edit_ft")

                    st.markdown("#### Financial Information")
                    new_salary = st.number_input("Salary ($) *", min_value=0, max_value=100000000, value=int(player_info['Salary']), step=100000, key="edit_salary")

                    st.markdown("---")
                    edit_submitted = st.form_submit_button("Save Changes", use_container_width=True)

                    if edit_submitted:
                        # Validate
                        if not new_player_name or new_player_name.strip() == "":
                            st.error("Please enter a player name.")
                        elif new_player_name.strip() != edit_player and new_player_name.strip() in df['player_name'].values:
                            st.error(f"A player named '{new_player_name.strip()}' already exists in the dataset.")
                        else:
                            # Create updated player data
                            updated_data = {
                                'player_name': new_player_name.strip(),
                                'conference': new_conference,
                                'games_played': int(new_games_played),
                                'mp': float(new_mp),
                                'ppg': float(new_ppg),
                                'rpg': float(new_rpg),
                                'apg': float(new_apg),
                                'spg': float(new_spg),
                                'bpg': float(new_bpg),
                                'to': float(new_to),
                                'fg%': float(new_fg_pct),
                                '3p%': float(new_three_pct),
                                '2p%': float(new_two_pct),
                                'ft%': float(new_ft_pct),
                                'Salary': int(new_salary)
                            }

                            # Calculate new CEM ratings
                            final_cem, mvpcem = calculate_single_player_metrics(updated_data, df[df['player_name'] != edit_player])

                            try:
                                update_player_in_csv(edit_player, updated_data)
                                st.success(f"Player '{new_player_name}' has been updated successfully!")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("New Final CEM Score", f"{final_cem:.2f}")
                                with col2:
                                    st.metric("New MVPCEM Score", f"{mvpcem:.2f}")

                                st.info("Please refresh the page to see the updated rankings.")
                                st.cache_data.clear()
                            except Exception as e:
                                st.error(f"Error updating player: {str(e)}")

        with manage_tab2:
            st.markdown("### Delete Player")

            # Player selection for deletion
            selected_player = st.selectbox(
                "Select a player to delete",
                options=["-- Select a player --"] + player_list,
                key="delete_player_select"
            )

            if selected_player != "-- Select a player --":
                # Show player details before deletion
                player_info = df[df['player_name'] == selected_player].iloc[0]

                st.markdown(f"**Selected Player:** {selected_player}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Conference", player_info['conference'])
                with col2:
                    st.metric("PPG", f"{player_info['ppg']:.1f}")
                with col3:
                    st.metric("RPG", f"{player_info['rpg']:.1f}")
                with col4:
                    st.metric("APG", f"{player_info['apg']:.1f}")

                st.warning(f"Are you sure you want to delete **{selected_player}** from the dataset? This action cannot be undone.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Delete Player", type="primary", use_container_width=True):
                        try:
                            delete_player_from_csv(selected_player)
                            st.success(f"Player '{selected_player}' has been deleted from the dataset.")
                            st.info("Please refresh the page to see the updated player list.")
                            st.cache_data.clear()
                        except Exception as e:
                            st.error(f"Error deleting player: {str(e)}")
                with col2:
                    if st.button("Cancel", use_container_width=True):
                        st.info("Deletion cancelled.")

if __name__ == "__main__":
    main()

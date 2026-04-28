import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv(r"allegations.csv")

# Helper for substantiated
def is_substantiated(disp):
    if pd.isna(disp):
        return False
    return str(disp).lower().startswith('substantiated')

df['substantiated'] = df['board_disposition'].apply(is_substantiated)

# ============================================================================
# PAGE 1: "A FEW OFFICERS ARE THE PROBLEM"
# ============================================================================

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle('PERSPECTIVE A: A Small Group of Officers Causes Most Misconduct', 
              fontsize=16, fontweight='bold', y=0.98)

# ----- Chart 1: Histogram - Distribution of substantiated complaints per officer -----
ax1 = axes1[0]

sub_counts = df[df['substantiated']].groupby('unique_mos_id').size()

ax1.hist(sub_counts, bins=range(1, int(sub_counts.max()) + 2),
         color='#D32F2F', edgecolor='black', alpha=0.7, rwidth=0.85)
ax1.set_xlabel('Number of Substantiated Complaints', fontsize=11)
ax1.set_ylabel('Number of Officers', fontsize=11)
ax1.set_title('Distribution: Substantiated Complaints per Officer', fontsize=13, fontweight='bold')
ax1.set_xticks(range(1, int(sub_counts.max()) + 1, 2))


# ----- Chart 2: Stacked Bar Chart - Top 10 officers: substantiated vs not sustained -----
ax2 = axes1[1]

# Get top 10 officers by total complaints
officer_total = df.groupby('unique_mos_id').size().sort_values(ascending=False).head(10).index
top10_df = df[df['unique_mos_id'].isin(officer_total)]

# Create pivot table for stacked bar
stacked_data = top10_df.groupby(['unique_mos_id', 'substantiated']).size().unstack(fill_value=0)
stacked_data.columns = ['Not Sustained', 'Substantiated']
stacked_data = stacked_data.sort_values('Substantiated', ascending=False)

stacked_data.plot(kind='bar', stacked=True, ax=ax2, color=['#BDBDBD', '#D32F2F'], width=0.8)
ax2.set_xlabel('Officer ID', fontsize=11)
ax2.set_ylabel('Number of Complaints', fontsize=11)
max_height = stacked_data.sum(axis=1).max()
ax2.set_ylim(0, max_height * 1.4)
ax2.set_title('Top 10 Officers: Complaint Outcomes', fontsize=13, fontweight='bold')
ax2.set_xticklabels([f'#{id}' for id in stacked_data.index], rotation=45, ha='right')
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

# Add percentage labels
for i, (idx, row) in enumerate(stacked_data.iterrows()):
    total = row['Substantiated'] + row['Not Sustained']
    sub_pct = (row['Substantiated'] / total) * 100 if total > 0 else 0
    ax2.text(i, total + 1, f'{sub_pct:.0f}%', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()


# ============================================================================
# PAGE 2: "THE SYSTEM/CONTEXT IS THE ISSUE"
# ============================================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('OPPOSING PERSPECTIVE: Most Complaints Are Dismissed and Volume Varies by Precinct', 
              fontsize=16, fontweight='bold', y=0.98)

# ----- Chart 3: Line Graph - Complaint outcomes over time -----
ax3 = axes2[0]

# Prepare yearly data
df['year'] = pd.to_numeric(df['year_received'], errors='coerce')
yearly_data = df[df['year'].between(2005, 2018)].groupby('year').agg({
    'complaint_id': 'count',
    'substantiated': 'sum'
})
yearly_data['not_sustained'] = yearly_data['complaint_id'] - yearly_data['substantiated']
yearly_data['sub_rate'] = (yearly_data['substantiated'] / yearly_data['complaint_id']) * 100

# Plot lines
ax3.plot(yearly_data.index, yearly_data['complaint_id'], 
         linewidth=2.5, color='#333333', marker='o', markersize=5, label='Total Complaints')
ax3.plot(yearly_data.index, yearly_data['substantiated'], 
         linewidth=2.5, color='#D32F2F', marker='s', markersize=5, label='Substantiated')
ax3.plot(yearly_data.index, yearly_data['not_sustained'], 
         linewidth=2.5, color='#757575', marker='^', markersize=5, label='Not Sustained')

ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Number of Complaints', fontsize=11)
ax3.set_title('Complaint Trends Over Time', fontsize=13, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# ----- Chart 4: Grouped Bar Chart - Top 8 precincts by complaint volume -----
ax4 = axes2[1]

# Get precinct data
df['precinct_clean'] = df['precinct'].astype(str).str.extract(r'(\d+)')[0]
precinct_data = df[df['precinct_clean'].notna()].groupby('precinct_clean').agg({
    'complaint_id': 'count',
    'substantiated': 'sum'
})
precinct_data['not_sustained'] = precinct_data['complaint_id'] - precinct_data['substantiated']
precinct_data = precinct_data.sort_values('complaint_id', ascending=False).head(8)

# Create grouped bar chart
x = np.arange(len(precinct_data))
width = 0.35

bars1 = ax4.bar(x - width/2, precinct_data['substantiated'], width, color='#D32F2F', label='Substantiated')
bars2 = ax4.bar(x + width/2, precinct_data['not_sustained'], width, color='#BDBDBD', label='Not Sustained')

ax4.set_xlabel('Precinct', fontsize=11)
ax4.set_ylabel('Number of Complaints', fontsize=11)
ax4.set_title('Complaint Outcomes by Precinct (Top 8)', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'Pct {p}' for p in precinct_data.index])
ax4.legend(loc='upper right')

plt.tight_layout()
plt.show()
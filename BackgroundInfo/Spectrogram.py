#Note: this was just my initial investigation. It's almost identical to the content in the matching jupyter notebook except that it also does a plot of 
# the signal just in the time domain at the end. I was just curious to see what the seal calls looked like in the time domain 

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

#path='../Guttural rupe' 
#File index
#File index
file =  '..\\..\\GreySealData\\Rupes A and B\\5713.210809120002'  #from PPT at time 892-896 (Rupe B)
#file = '..\\..\\GreySealData\\Guttural rupe\\5711.211013040024'
#file = '..\\..\\GreySealData\\Rupes A and B\\5713.210825190002'
#file =          '..\\..\\GreySealData\\Moan\\5713.210902110002'  #from PPT at time 212 seconds

# File path
annot_file_path = file +'.Table.1.selections.txt'

# Reading the file into a DataFrame
df = pd.read_csv(annot_file_path, sep='\t')

# Display the first few rows of the DataFrame
print(df.head())

sample_rate, samples = wavfile.read(file+'.wav')

print(sample_rate)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=2456, nfft=4096, noverlap=1228, window='hann')
spectrogram[spectrogram < 0.001] = 0.001
print(np.shape(frequencies))
print((frequencies))
print("Shape: ", np.shape(spectrogram))

"""
flattened_spectrogram = spectrogram.flatten()
plt.hist(flattened_spectrogram, bins=20, edgecolor='black')  # Adjust 'bins' as needed
plt.title("Histogram of Spectrogram Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
"""
fmin = 0 # Hz
fmax = 1000 # Hz
freq_slice = np.where((frequencies >= fmin) & (frequencies <= fmax))

# keep only frequencies of interest
frequencies = frequencies[freq_slice]
spectrogram = spectrogram[freq_slice,:][0]


pc = plt.pcolormesh(times, frequencies, spectrogram, norm=LogNorm(), cmap='Spectral_r')
#pc = plt.pcolormesh(times, frequencies, np.log(spectrogram))
cbar = plt.colorbar(pc)
#plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

#Define colors for annotations
annotation_colors = {
    "Rupe A": "red",
    "Rupe B": "green",
    "Growl B": "yellow",
    "Rupe C" : "purple",
    "Moan": "pink",
    "G rupe" : "blue"
}

def overlay_annotations(ax, df, annotation_colors):
    #Track labels to ensure they are added only once in the legend
    added_labels = set()

    for _, row in df.iterrows():
        start_time = row['Begin Time (s)']
        end_time = row['End Time (s)']
        low_freq = row['Low Freq (Hz)']
        high_freq = row['High Freq (Hz)']
        annotation = row['Annotation']

        #Skip if the annotation is not in the defined colors
        if annotation not in annotation_colors:
            continue

        #Draw rectangles
        ax.add_patch(
            plt.Rectangle(
                (start_time, low_freq),  #Bottom-left corner
                end_time - start_time,  #Width (time)
                high_freq - low_freq,  #Height (frequency)
                edgecolor=annotation_colors[annotation],
                facecolor='none',
                linewidth=2,
                label=annotation if annotation not in added_labels else None  #Add label once
            )
        )
        added_labels.add(annotation)  #Mark label as added

    #Add legend
    ax.legend(loc='upper right')




def update_colormap(event):
    #Get current view limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    #Find indices corresponding to the current view limits
    x_indices = np.where((times >= xlim[0]) & (times <= xlim[1]))[0]
    y_indices = np.where((frequencies >= ylim[0]) & (frequencies <= ylim[1]))[0]

    #Handle cases where no data is visible
    if len(x_indices) == 0 or len(y_indices) == 0:
        return

    #Extract the visible data
    data_visible = spectrogram[np.ix_(y_indices, x_indices)]
    #data_visible = np.log(spectrogram)[np.ix_(y_indices, x_indices)]

    #Compute new color limits
    vmin = np.nanmin(data_visible)
    vmax = np.nanmax(data_visible)

    #Update the color limits of the pcolormesh
    pc.set_clim(vmin=vmin, vmax=vmax)
    
    #Update the colorbar to reflect the new color limits
    cbar.update_normal(pc)

    #Redraw the figure
    plt.draw()
    

#Get the current axes
ax = plt.gca()
overlay_annotations(ax, df, annotation_colors)

#Connect the update function to the axes limit change events
ax.callbacks.connect('xlim_changed', update_colormap)
ax.callbacks.connect('ylim_changed', update_colormap)

plt.show()
#plt.show()
############################################################


Time = np.linspace(0, len(samples) / sample_rate, num=len(samples))
print(np.shape(Time))

#audio = input_data[1]  only for stereo
#print(audio)
# plot the first 1024 samples
plt.plot(Time, samples)
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title  
plt.title("Sample Wav")
# display the plot

#plt.show()

# Overlay bounding boxes
for index, row in df.iterrows():
    start_time = row['Begin Time (s)']
    end_time = row['End Time (s)']
    low_freq = row['Low Freq (Hz)']
    high_freq = row['High Freq (Hz)']
    
    # Draw a rectangle
    plt.gca().add_patch(
        plt.Rectangle(
            (start_time, low_freq),  # Bottom-left corner
            end_time - start_time,  # Width (time)
            high_freq - low_freq,  # Height (frequency)
            edgecolor='red',
            facecolor='none',
            linewidth=2,
            label=row['Annotation'] if index == 0 else None  # Label only once
        )
    )

plt.legend(loc='upper right')
plt.show()
from Detector.database import *
from Detector.model import *
from Detector.featureSelection import *
from play_ground import filter_articles
from Detector.detect import *
from articleProcessing import *

my_private_trove_key = my_private_trove_key = "3j228nhbi1pftav2"

start = time.time()

thesis_0 = ['avg star frequency', '"wind" tfidf value', '"cyclone" tfidf value', '"damage" tfidf value', '"storm" tfidf value', '"last" tfidf value', '"rain" tfidf value,', '"heavy" tfidf value', '"yesterday" tfidf value', '"night" tfidf value', '"town" tfidf value', '"wind" frequency', '"heavy" frequency', 'five star frequency']
thesis_1 = ['avg star frequency', '"wind" tfidf value', '"cyclone" tfidf value', '"damage" tfidf value', '"storm" tfidf value', '"last" tfidf value', '"rain" tfidf value,', '"heavy" tfidf value', '"yesterday" tfidf value', '"night" tfidf value', '"town" tfidf value']
thesis_2 = ['avg star frequency']
# build_dataset('Tropical Cyclone', 4, '1930-01-01', '1940-01-01', 'perilAUS', 'events', 'filtered-database', '1930-1939', ['cyclone', 'damage', 'storm', 'heavy', 'wind', 'rain', 'last', 'night', 'yesterday', 'town'], ['heavy', 'wind', 'cyclone'], True, True, r'test filtered 1930-1939 train.pkl', True)

# print(len(get_peril_dates('Tropical Cyclone', '1940-01-01', '1950-01-01', 'perilAUS', 'events')))

# create_confidence_interval(test_model(keep_columns(load_df('filtered 1930-1939 all.pkl'), ['avg star frequency']), r'1910-1919 avg star model 2.sav', True), 0.05, 'confidence intervals 1930-1939.pkl')
# return_ids(filter_known_events(group_events(load_df('confidence intervals 1930-1939.pkl')), 'perilAUS', 'events', '1930-01-01', '1940-01-01'), 'filtered-database', '1930-1939', 0.5, 3)

# create_confidence_interval(test_model(keep_columns(load_df('filtered 1900-1909 all.pkl'), ['avg star frequency']), r'1910-1919 avg star model 2.sav', True), 0.2, 'confidence intervals 1900-1909.pkl')
# return_ids(filter_known_events(group_events(load_df('confidence intervals 1900-1909.pkl')), 'perilAUS', 'events', '1900-01-01', '1910-01-01'), 'filtered-database', '1900-1909', 0.5, 3)

# create_confidence_interval(test_model(keep_columns(load_df('filtered 1940-1949 all.pkl'), ['avg star frequency']), r'1910-1919 avg star model 2.sav', True), 0.2, 'confidence intervals 1940-1949.pkl')
# return_ids(filter_known_events(group_events(load_df('confidence intervals 1940-1949.pkl')), 'perilAUS', 'events', '1940-01-01', '1950-01-01'), 'filtered-database', '1940-1949', 0.5, 3)

# plot_df(keep_columns(load_df('filtered 1940-1949 all.pkl'), ['article frequency']), True)

# create_confidence_interval(test_model(keep_columns(load_df('filtered 1940-1949 all.pkl'), thesis_2), r'KNN 1910-1919 all model.sav', True), 0.4, 'KNN thesis2 confidence intervals 1940-1949.pkl')
# print(return_ids(filter_known_events(group_events(load_df('KNN thesis2 confidence intervals 1940-1949.pkl')), 'perilAUS', 'events', '1940-01-01', '1950-01-01'), 'filtered-database', '1940-1949', 0.9, 3).to_string())

# create_confidence_interval(test_model(load_df('filtered 1940-1949 all.pkl'), r'filtered 1910-1919 all model.sav', True), 0.4, 'confidence intervals 1940-1949.pkl')
# return_ids(filter_known_events(group_events(load_df('confidence intervals 1940-1949.pkl')), 'perilAUS', 'events', '1940-01-01', '1950-01-01'), 'filtered-database', '1940-1949', 0.5, 3)

# create_confidence_interval(test_model(load_df('filtered 1940-1949 all.pkl'), r'filtered 1910-1919 all model.sav', True), 0.4, 'confidence intervals 1940-1949.pkl')
# return_ids(filter_known_events(group_events(load_df('confidence intervals 1940-1949.pkl')), 'perilAUS', 'events', '1940-01-01', '1950-01-01'), 'filtered-database', '1940-1949', 0.5, 3)


# train(keep_columns(load_df('filtered 1910-1919 all.pkl'), thesis_2), 'SVM 1910-1919 all model.sav', 'SVM')
# test_model(load_df('filtered 1940-1949 all.pkl'), '1910-1919 model.sav', True)

# train(keep_columns(load_df('filtered 1910-1919 all.pkl'), thesis_2), 'RF 1910-1919 all model.sav', 'RF')
# test_model(keep_columns(load_df('filtered 1940-1949 all.pkl'), thesis_2), 'RF 1910-1919 all model.sav', True)

# train(keep_columns(load_df('filtered 1910-1919 all.pkl'), thesis_2), 'KNN 1910-1919 all model.sav', 'KNN')
# test_model(keep_columns(load_df('filtered 1940-1949 all.pkl'), thesis_2), 'KNN 1910-1919 all model.sav', True)

# plot_df(load_df('filtered 1900-1909 all.pkl'), True)


# df = create_ranker_df('1900-01-01', '1910-01-01', 'article-database', '1900-1910')

# print(filter_articles('1900-01-01', '1901-01-01', 'article-database', '1900-1910', 'filtered 1900-1901.pkl'))

# filter_and_insert('1900-01-01', '1910-01-01', 'article-database', '1900-1909', 'filtered-database', '1900-1909')

# plot_df(load_df(r'14 days\filtered 1940-1949 all.pkl'), True)
# print(load_df('filtered 1900-1901.pkl').to_string())


# print(load_df('1911-1915 all.pkl'))

# process_data(get_data_between_dates('1900-01-01', '1900-01-15', 'article-database', '1900-1903'))

# find_important_keywords('Tropical Cyclone', 4, '1930-01-01', '1939-12-01', 'perilAUS', 'events', 'filtered-database', '1930-1939')

# print(get_tfidf_in_proximity('1900-01-01', '1905-01-01', 'article-database', '1900-1910', 14, 'cyclone'))

# correlate_frequency_event_occurrence('Tropical Cyclone', 14, '1900-01-01', '1906-06-01', 'perilAUS', 'events', 'article-database', '1900-1910', 'cyclone', True, True)

# build_dataset('Tropical Cyclone', 4, '1900-01-01', '1910-01-01', 'perilAUS', 'events', 'filtered-database', '1900-1909', ['cyclone', 'damage', 'storm', 'heavy', 'wind', 'rain', 'last', 'night', 'yesterday', 'town'], ['heavy', 'wind'], True, True, r'filtered 1900-1909 all.pkl', False)
# build_dataset('Tropical Cyclone', 4, '1910-01-01', '1920-01-01', 'perilAUS', 'events', 'filtered-database', '1910-1919', ['cyclone', 'damage', 'storm', 'heavy', 'wind', 'rain', 'last', 'night', 'yesterday', 'town'], ['heavy', 'wind'], True, True, r'filtered 1910-1919 train.pkl', True)
# build_dataset('Tropical Cyclone', 4, '1920-01-01', '1930-01-01', 'perilAUS', 'events', 'filtered-database', '1920-1929', ['cyclone', 'damage', 'storm', 'heavy', 'wind', 'rain', 'last', 'night', 'yesterday', 'town'], ['heavy', 'wind'], True, True, r'filtered 1920-1929 all.pkl', False)
# build_dataset('Tropical Cyclone', 4, '1930-01-01', '1940-01-01', 'perilAUS', 'events', 'filtered-database', '1930-1939', ['cyclone', 'damage', 'storm', 'heavy', 'wind', 'rain', 'last', 'night', 'yesterday', 'town'], ['heavy', 'wind'], True, True, r'test filtered 1930-1939 all.pkl', False)
# build_dataset('Tropical Cyclone', 4, '1940-01-01', '1950-01-01', 'perilAUS', 'events', 'filtered-database', '1940-1949', ['cyclone', 'damage', 'storm', 'heavy', 'wind', 'rain', 'last', 'night', 'yesterday', 'town'], ['heavy', 'wind'], True, True, r'filtered 1940-1949 train.pkl', True)


# print(get_tfidf_in_proximity('1900-01-01', '1900-06-01', 'article-database', '1900-1910', 14))

# print(get_word_frequency_per_day('1902-09-14', '1902-09-15', 'article-database', '1900-1903', 14)[1][2])

# print(len(get_data_between_dates('1919-01-01', '1920-01-01', 'filtered-database', '1910-1919')), 'found')

# extract_and_insert(my_private_trove_key, 'cyclone', '1900', '1909', 'article-database', '1900-1909')

# sample_text = "Dios mio one two my amigo A wonderful serenity has taken possession of my entire soul, like these sweet mornings of spring which I enjoy with my whole heart. I am alone, and feel the charm of existence in this spot, which was created for the bliss of souls like mine. I am so happy, my dear friend, so absorbed in the exquisite sense of mere tranquil existence, that I neglect my talents. I should be incapable of drawing a single stroke at the present moment; and yet I feel that I never was a greater artist than now. When, while the lovely valley teems with vapour around me, and the meridian sun strikes the upper surface of the impenetrable foliage of my trees, and but a few stray gleams steal into the inner sanctuary, I throw myself down among the tall grass by the trickling stream; and, as I lie close to the earth, a thousand unknown plants are noticed by me: when I hear the buzz of the little world among the stalks, and grow familiar with the countless indescribable forms of the insects and flies, then I feel the presence of the Almighty, who formed us in his own image, and the breath"
# samp_text = "<html> 281731 1231 000 0102301 0 holy-hell, asde feasdz bigasndja this is the ! strange st, thing, i've BALLS strange HOLY @#! ever seen! Dr. Michael is waiting in lab number five on 15 Clarence street. </html>"
samp_text = "<p><span> Damage donebyihe</span><span> cyclone which struck</span><span> Barigglow on Friday</span><span> night is estimated at</span></p> <p><span> Between £30,000 and</span><span> £40,000.</span><span> The outskirts of the town were</span><span> still without light last night.</span>"
print(pre_process(samp_text))

end = time.time()

print('Overall time taken:', end - start)

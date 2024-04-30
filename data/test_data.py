
import numpy as np
import pandas as pd
import random
from langdetect import detect


#---------------------------------------------------------
#data for redacting
#---------------------------------------------------------

test_list_review = [
            'John was great at showing us around', 
            'The receptionist, Sophia, was so welcoming',
            "I can't believe how lovely Florence was.",
            "Mohammed was so helpful"
            ]


#---------------------------------------------------------
#Create dummy data set of text reviews
#---------------------------------------------------------
def create_dummy_dataset():
    list_of_positive_experiences = [

    # Staff
    "The staff were incredibly friendly and helpful. They went above and beyond to make sure I was comfortable and well-informed.",
    "The nurses were very compassionate and understanding. They took the time to listen to my concerns and answer my questions thoroughly.",
    "The doctor was extremely knowledgeable and experienced. He/She took the time to explain my condition and treatment options in a way that I could understand.",
    "The entire staff was very professional and efficient. They worked together to make sure I was seen quickly and that my needs were met.",
    "I felt very respected and valued by all of the staff. They made me feel like I was their only patient.",
    "The staff was very attentive to my needs. They were always there to answer my questions and provide comfort.",
    "I was very impressed with the level of care I received from the staff. They were all highly skilled and experienced.",
    "The staff was very supportive and encouraging. They helped me to stay positive and motivated throughout my appointment.",
    "I felt very safe and secure in the care of the staff. They were always there to look after me and make sure I was okay.",
    "I am so grateful for the care I received from the staff. They went above and beyond to make sure I had a positive experience.",

    # Cleanliness
    "The entire facility was very clean and well-maintained.",
    "The examination rooms were spotless and well-organized.",
    "The staff took great care to clean and disinfect all surfaces between patients.",
    "I felt very safe and comfortable in the facility, knowing that it was clean and hygienic.",
    "The waiting room was clean and tidy, with plenty of comfortable seating.",
    "The restrooms were clean and well-stocked.",
    "I was impressed with the overall cleanliness of the facility.",
    "I could tell that the staff takes great pride in keeping the facility clean and safe for patients.",
    "I felt confident that I was in good hands, knowing that the facility was clean and well-maintained.",
    "I am grateful for the staff's commitment to cleanliness and hygiene."
    ]
    
    list_of_negative_experiences = [

    # Waiting times
    "I had to wait over an hour to see the doctor.",
    "The waiting room was crowded and uncomfortable.",
    "I had to reschedule my appointment because I couldn't find parking.",
    "I missed a meeting at work because I was stuck in traffic on the way to my appointment.",
    "I felt like my time was not valued by the staff. I had to wait a long time for everything, from checking in to seeing the doctor to getting my prescriptions filled.",
    "I was very frustrated by the long wait times. I felt like I was wasting my time.",
    "The staff didn't seem to care that I was waiting a long time. They didn't apologize for the delay or offer me any updates.",
    "I felt like the staff was more concerned with their own schedules than with the needs of the patients.",
    "I would not recommend this healthcare facility to others because of the long wait times.",
    "I had a very negative experience at this healthcare facility because of the long wait times.",

    # Car parking
    "The car parking at the healthcare facility was limited and expensive.",
    "I had to circle the car park for 15 minutes before I could find a space.",
    "I had to walk a long way from the car park to the healthcare facility.",
    "I had to pay a high fee to park at the healthcare facility.",
    "I felt like the healthcare facility was trying to make money off of patients by charging high parking fees.",
    "I was very frustrated by the difficulty of finding parking at the healthcare facility.",
    "The staff didn't seem to care that the car parking was so difficult. They didn't offer any assistance to patients who were struggling to find a space.",
    "I felt like the healthcare facility was not making an effort to make it easy for patients to get to their appointments.",
    "I would not recommend this healthcare facility to others because of the difficulty of finding parking.",
    "I had a very negative experience at this healthcare facility because of the difficulty of finding parking."
    ]

    #other languages 
    german_positive_reviews = [
    "Tolles Hotel, freundliches Personal und exzellentes Essen.",
    "Tolle Lage, in der Nähe vieler Sehenswürdigkeiten.",
    "Schönes Ambiente und gemütliche Zimmer.",
    "Gutes Preis-Leistungs-Verhältnis und leckeres Frühstück.",
    "Das Personal war sehr zuvorkommend und hilfsbereit.",
    ]

    german_negative_reviews = [
        "Zimmer war dreckig und das Badezimmer hatte Schimmel.",
        "Der Service war unfreundlich und inkompetent.",
        "Lärm von der Straße störte unseren Aufenthalt.",
        "Das WLAN war langsam und unzuverlässig.",
        "Das Hotelzimmer roch nach Zigarettenrauch."
    ]

    spanish_positive_reviews = [
        "El personal del hotel es amable y atento.",
        "Buena ubicación, cerca de la playa y restaurantes.",
        "El desayuno era delicioso y variado.",
        "El precio es asequible y las habitaciones son cómodas.",
        "Excelente atención del personal en el restaurante."
    ]

    spanish_negative_reviews = [
        "La habitación estaba sucia y no había agua caliente.",
        "El servicio de limpieza dejó mucho que desear.",
        "La piscina estaba sucia y el agua fría.",
        "Problemas con el aire acondicionado en la habitación.",
        "Ruido constante de la construcción cercana."
    ]

    polish_positive_reviews = [
        "Hotel ma świetną lokalizację i widok na morze.",
        "Pokoje są czyste i zadbane.",
        "Smaczne śniadanie wliczone w cenę noclegu.",
        "Dogodna cena i dostępność restauracji w okolicy.",
        "Przesympatyczna obsługa w hotelowej restauracji."
    ]

    polish_negative_reviews = [
        "Obsługa była nieprofesjonalna i nieuprzejma.",
        "Brak ciepłej wody pod prysznicem.",
        "Basen był zamknięty bez informacji.",
        "Kiepska izolacja dźwiękowa w pokojach.",
        "Utrudniony dojazd do hotelu ze względu na remonty."
    ]

    punjabi_positive_reviews = [
    "ਮੇਰਾ ਸਵਾਗਤ ਹਰ ਵਾਰ ਖੁਸ਼ੀ ਦੇ ਸਾਥ ਹੁੰਦਾ ਹੈ। ਡਾਕਟਰ ਸਾਬ ਸਾਰੇ ਸਵਾਲਾਂ ਦਾ ਉਤਤਰ ਦੇਂਦੇ ਹਨ।",
    "ਮੈਂ ਇੱਕ ਬਹੁਤ ਵਧੀਆ ਹੇਲਥ ਐਪਾਇੰਟਮੈਂਟ ਦਾ ਅਨੁਭਵ ਕੀਤਾ, ਸਾਰਾ ਸਟਾਫ ਦਿਲ ਦੀ ਪਸੰਦ ਹੈ।",
    "ਮੇਰੀ ਸਾਰੀ ਸਮੱਸਿਆਵਾਂ ਨੂੰ ਠੀਕ ਕਰ ਦਿੱਤਾ ਗਿਆ ਅਤੇ ਮੈਂ ਸਹੀ ਤਰ੍ਹਾਂ ਇਲਾਜ ਹੋ ਗਿਆ।",
    "ਡਾਕਟਰ ਦੀ ਸਲਾਹ ਨੇ ਮੇਰੀ ਸਿਹਤ ਨੂੰ ਵਧੇਰਾ ਕੀਤਾ, ਸਹੀ ਇਲਾਜ ਕਰਦੇ ਸਮੇਂ ਦੀ ਸਹਾਇਕ ਸੀ।",
    "ਇਸ ਸਵਾਗਤ ਅਤੇ ਇਲਾਜ ਦੇ ਵਕਤ ਮੇਰੀ ਰਾਹਤ ਮਿਲੀ, ਅਤੇ ਮੈਂ ਸਾਰੇ ਡਾਕਟਰ ਸਟਾਫ ਨੂੰ ਧੰਨਵਾਦ ਦੇਂਦਾ ਹਾਂ।",
    "ਮੇਰੇ ਇਲਾਜ ਦੇ ਪ੍ਰਗਤਾਂ ਨੂੰ ਵੱਧ ਦਿੱਤਾ ਗਿਆ ਅਤੇ ਮੈਂ ਹੋਰ ਠੋਸ ਹੋ ਗਿਆ।",
    "ਮੇਰੀ ਸਿਹਤ ਦਾ ਖ਼ਿਯਾਲ ਵਧ ਗਿਆ ਅਤੇ ਮੈਂ ਡਾਕਟਰ ਦੀ ਸਲਾਹ ਦੀ ਸਨ੍ਮਾਨ ਕਰਦਾ ਹਾਂ।",
    "ਇਸ ਸਾਰੇ ਵੇਲ਼ੇ ਅਤੇ ਇਲਾਜ ਦੇ ਦੌਰਾਨ, ਸਾਰਾ ਸਟਾਫ ਨੇ ਸਾਨੂੰ ਸਬ ਖੁਸ਼ ਰੱਖਿਆ।",
    "ਮੈਂ ਇਸ ਸ੍ਰੀ ਹੇਲਥਕੇਅਰ ਸੈਂਟਰ ਦੀ ਸੇਵਾ ਨੂੰ ਪੂਰੀ ਤਰ੍ਹਾਂ ਦਾ ਅਤੇ ਪੂਰੀ ਦਿਲ ਨਾਲ ਪਸੰਦ ਕਰਦਾ ਹਾਂ।",
    ]

    #build combined df with all language reviews present to test pie chart visualisation.
    list_positive_reviews_other_lang = [
        german_positive_reviews,
        polish_positive_reviews,
        spanish_positive_reviews,
        punjabi_positive_reviews
    ]

    list_negative_reviews_other_lang = [
        german_negative_reviews,
        polish_negative_reviews,
        spanish_negative_reviews
    ]

    for list_lang_reviews in list_positive_reviews_other_lang:
        for review in list_lang_reviews:
            list_of_positive_experiences.append(review)

    for list_lang_reviews in list_negative_reviews_other_lang:
        for review in list_lang_reviews:
            list_of_negative_experiences.append(review)

    list_fft_labels = [
    'very good',
    'good',
    'neither good nor poor',
    'poor',
    'very poor'
    ]

    random.seed(42)
    list_fft_score_good_reviews = [random.choice(list_fft_labels[:2]) for review in range(len(list_of_positive_experiences))]

    random.seed(42)
    list_fft_score_bad_reviews = [random.choice(list_fft_labels[3:]) for review in range(len(list_of_negative_experiences))]

    list_combined_reviews = []
    list_combined_fft_scores = []

    for review in list_of_positive_experiences:
        list_combined_reviews.append(review)

    for review in list_of_negative_experiences:
        list_combined_reviews.append(review)

    for score in list_fft_score_good_reviews:
        list_combined_fft_scores.append(score)

    for score in list_fft_score_bad_reviews:
        list_combined_fft_scores.append(score)
    
    #randomly assign gender to test data
    random.seed(42)
    list_gender = [random.choice(['Male', 'Female']) for review in range(len(list_combined_reviews))]
    #randomly assign ethnicity to test data
    random.seed(42)
    list_ethnicity = [random.choice(['White', 'Mixed', 'Asian', 'Black', 'Other']) for review in range(len(list_combined_reviews))]
    #randomly assign a service label to test data
    random.seed(42)
    list_service = [random.choice(['Service A', 'Service B', 'Service C']) for review in range(len(list_combined_reviews))]



    data = [{
        "Review": list_combined_reviews[i], 
        "fft_score": list_combined_fft_scores[i],
        "gender": list_gender[i],
        "ethnicity": list_ethnicity[i],
        "service": list_service[i]} for i in range(len(list_combined_fft_scores))]
    
    df_fft_reviews = pd.DataFrame(data)

    #update the source data to have a single binary ground truth column
    string1 = 'very good'
    string2 = 'good'

    #original line for splitting into binary labels
    #df_fft_reviews['fft_binary'] = np.where((df_fft_reviews['fft_score'] == string1) | (df_fft_reviews['fft_score'] == string2), 'Positive', 'Negative')

    return df_fft_reviews


#------------------------------------------------------

def create_dummy_survey_responses():
    """
    Function to create a df of dummy survey responses to a given question.

    The question used is: "provide examples of what your organisation currently 
    does in terms of system thinking and partnership working in relation to: 
    Co-producing vision and values with other system partners"

    Returns df of these dummy responses.
    """
    #Theme 1: Engaging with system partners to develop a shared vision and values
    list_theme_1 = [
    "Our organization works closely with other system partners to develop and implement a shared vision and values for the region. We meet regularly to discuss our goals and priorities, and we collaborate on projects and initiatives to achieve them.",
    "We have established a joint forum with our system partners where we can share information, best practices, and lessons learned. We also use the forum to develop consensus on important issues and to identify opportunities for collaboration.",
    "We work with our system partners to develop and implement shared performance measures, so that we can track our progress towards our shared goals. We also use this information to identify areas where we need to improve our collaboration.",
    "We support our system partners to develop their own capacity for system thinking and partnership working. We offer training and workshops, and we provide access to resources and expertise.",
    "We have established a system-wide vision and values development process that involves all system partners. This process includes regular meetings, workshops, and consultations.",
    "We have developed a shared system dashboard that tracks key performance indicators related to our vision and values. This dashboard is shared with all system partners so that we can track our progress together.",
    "We have established a system-wide innovation fund that supports system partners to develop and implement new projects and initiatives that align with our vision and values.",
    "We have created a system-wide learning and development program that helps system partners to develop the skills and knowledge they need to achieve our shared vision and values.",
    "We have established a system-wide communication and engagement plan that helps us to communicate our vision and values to stakeholders and to engage them in the co-production of our services and programs."
    ]

    #Theme 2: Involving stakeholders in the co-production of vision and values
    list_theme_2 = [
    "We involve stakeholders in the co-production of our vision and values through a variety of mechanisms, including surveys, consultations, and focus groups. We also have a stakeholder council that meets regularly to advise us on our strategic planning and decision-making.",
    "We involve stakeholders in the development and implementation of our policies and procedures by providing them with opportunities to comment on draft documents and to participate in public hearings. We also establish working groups with stakeholder representatives to develop specific policies and procedures.",
    "We provide opportunities for stakeholders to participate in our projects and initiatives by establishing volunteer programs, advisory groups, and community engagement programs. We also offer funding opportunities to stakeholders to support their own projects and initiatives that align with our vision and values.",
    "We communicate our vision and values to stakeholders on a regular basis through our website, social media, and print and electronic publications. We also hold public meetings and workshops to share our vision and values and to answer stakeholder questions.",
    "We seek stakeholder feedback on our vision and values through surveys, focus groups, and public meetings. We use this feedback to improve our communication and engagement efforts, and to ensure that our vision and values reflect the needs and priorities of our stakeholders.",
    "We have established a stakeholder engagement forum that meets regularly to discuss our vision and values. This forum is open to all stakeholders, and it provides them with an opportunity to share their feedback and ideas.",
    "We have developed a stakeholder engagement toolkit that provides resources and guidance to system partners on how to engage stakeholders in the co-production of vision and values.",
    "We have established a stakeholder engagement policy that outlines our commitment to engaging stakeholders in all aspects of our work.",
    "We track and report on our stakeholder engagement activities so that we can identify areas for improvement.",
    "We provide training and support to system partners on how to engage stakeholders in the co-production of vision and values."
    ]

    combined_list = list_theme_1 + list_theme_2

    # Set a seed value to ensure reproducibility
    seed_value = 42
    random.seed(seed_value)

    # Generate a list of labels "M" and "F" with the same length as the reference list
    gender_list = [random.choices(['M', 'F'])[0] for i in range(len(combined_list))]

    df = pd.DataFrame(data=[combined_list, gender_list]).T 
    df.rename(columns={0: "Response", 1: "Gender"}, inplace=True)
    return df


#--------------------------------------------------------------

#---------------------------------------------------------
#Create dummy data set of text reviews
#---------------------------------------------------------
def create_dummy_dataset_english_only():
    #Themes: caring staff, curing ailment, other
    list_of_positive_experiences = [
    #caring staff
    "Comfortable and informed surgery experience due to attentive, supportive staff.",
    "Rehabilitation center staff's patience, encouragement, and celebration of progress led to successful recovery.",
    "Therapist provided a safe space, empathy, and guidance for mental health healing.",
    "Children's hospital nurses treated a child with kindness and compassion, making them feel like a superhero.",
    "Gentle dentist and calming demeanor eased dental anxiety, resulting in a confident smile.",
    "Home care nurse's attentiveness and compassion ensured the best possible care at home for an elderly parent.",
    "Midwives created a peaceful and empowering birthing environment, making it an unforgettable experience.",
    "Cancer support group volunteers provided invaluable emotional support and shared experiences.",
    "Animal hospital staff treated a pet with the same care and concern as a human patient, saving its life.",
    "Pharmacist went the extra mile to find an affordable medication, making a significant difference.",
    
    #curing ailments
    "Minimally invasive surgery led to a quick recovery and minimal scarring, enabling a return to an active lifestyle.",
    "Targeted therapy brought control over a chronic illness, thanks to the doctor's innovative approach.",
    "Physical therapy program regained strength and mobility after an injury, with personalized exercises and constant support.",
    "Routine screening and prompt treatment by a diligent healthcare team saved a life from cancer.",
    "Mental health treatment helped overcome anxiety and depression, leading to a fulfilling life.",
    "Weight loss program provided tools and motivation, with a sustainable plan created by a nutritionist and fitness coach.",
    "Pain management clinic equipped a patient with strategies to manage chronic pain, reducing reliance on medication.",
    "Advanced diagnostic techniques revealed the cause of a mysterious illness, allowing for targeted treatment and recovery.",
    "Telemedicine appointment offered convenient access to a specialist, resolving a health concern without leaving home.",
    "Preventative care plan helped avoid a potential health problem, thanks to a proactive approach from a healthcare provider.",
    
    #other
    "Community health fair provided valuable information and resources, promoting overall well-being.",
    "Blood donation experience was quick and painless, with friendly staff making it a rewarding act.",
    "Urgent care center provided prompt and effective treatment for a sudden illness, preventing complications.",
    "Dental hygienist's thorough cleaning and helpful tips improved oral health significantly.",
    "Vision checkup identified a potential eye problem early, allowing for preventive measures.",
    "Allergy clinic diagnosed and treated allergies effectively, offering relief from seasonal discomfort.",
    "Speech therapist helped regain lost communication skills after an accident, restoring confidence and independence.",
    "Physical exam with a thorough doctor offered peace of mind and identified potential health risks early.",
    "Online patient portal provided convenient access to medical records and appointment scheduling.",
    "Follow-up care after a major surgery ensured a smooth recovery process and addressed any concerns."
    ]
    
    #themes: waiting times, not curing ailment
    list_of_negative_experiences = [

    # Waiting times
    "Endless waiting room: Spent 5 hours for a 15-minute appointment. Scheduling needs serious improvement.",
    "Lost in limbo: Stuck in the queue for pre-op tests, missed my surgery time, and rescheduled for next week.",
    "Delayed diagnosis: Waited months for test results, delaying vital treatment and causing unnecessary anxiety.",
    "Phone tag torture: Countless calls to reschedule, then waited an hour past the new appointment time.",
    "Parking purgatory: Finding a spot at the hospital took longer than the actual consultation.",
    "Appointment agony: Online booking promised convenience, but endless loading screens led to frustration.",
    "Children's chaos: Long waits in a cramped room with sick kids and screaming parents tested everyone's patience.",
    "Urgent care nightmare: 4-hour wait for a simple sprain, felt worse by the time I saw the doctor.",
    "Specialist saga: Referral sent months ago, still waiting for an appointment, health issue worsening in the meantime.",
    "Canceled care: Appointment canceled hours before, no explanation given, forced to reschedule weeks later.",

    # Car parking
    "Misdiagnosis marathon: Three doctors, three conflicting diagnoses, still no clear answer to my symptoms.",
    "Treatment treadmill: Medication changed repeatedly, side effects worsen, original problem remains unsolved.",
    "Painful progress: Physical therapy didn't improve my injury, left me feeling frustrated and discouraged.",
    "Surgery setback: Underwent a major surgery, but complications and pain persist, questioning its effectiveness.",
    "Emotional void: Mental health therapist felt disconnected, offered generic advice, no significant improvement.",
    "Diet dead end: Dietitian's personalized plan didn't work, weight loss stalled, feeling discouraged and helpless.",
    "Allergy agony: Tried multiple medications, allergy shots, immunotherapy, still dealing with constant symptoms.",
    "Chronic confusion: Specialist gave vague explanations, left me feeling uninformed and worried about my condition.",
    "Communication catastrophe: Doctor barely listened, rushed through the appointment, ignored my questions.",
    "Financial frenzy: Expensive tests, unnecessary procedures, left with a hefty bill and unresolved health concerns.",

    #other
    "Dismissive doctor: Felt rushed, unheard, and judged, left with no answers and a sense of hopelessness.",
    "Outdated equipment: Facilities felt run-down, equipment looked old and unreliable, questioned the quality of care.",
    "Lack of empathy: Staff seemed jaded and indifferent, felt like just another number, not a patient in need.",
    "Hygiene horrors: Unclean waiting room, dirty linens, instruments didn't appear properly sterilized.",
    "Accessibility obstacle: No ramps, narrow doorways, difficult for older or disabled patients to navigate.",
    "Medication mix-up: Wrong medication dispensed, almost taken it before realizing the mistake, potentially dangerous.",
    "Privacy panic: Medical records shared without my consent, felt my information was not secure.",
    "Follow-up fiasco: No follow-up calls, reminders, or appointments after surgery, left to manage recovery on my own.",
    "Billing blunder: Overcharged for services, insurance claim denied, dealing with billing department bureaucracy.",
    "Broken trust: Lost confidence in the healthcare system after multiple negative experiences, seeking alternative solutions."
    ]

    list_fft_labels = [
    'very good',
    'good',
    'neither good nor poor',
    'poor',
    'very poor'
    ]

    random.seed(42)
    list_fft_score_good_reviews = [random.choice(list_fft_labels[:2]) for review in range(len(list_of_positive_experiences))]

    random.seed(42)
    list_fft_score_bad_reviews = [random.choice(list_fft_labels[3:]) for review in range(len(list_of_negative_experiences))]

    list_combined_reviews = []
    list_combined_fft_scores = []

    for review in list_of_positive_experiences:
        list_combined_reviews.append(review)

    for review in list_of_negative_experiences:
        list_combined_reviews.append(review)

    for score in list_fft_score_good_reviews:
        list_combined_fft_scores.append(score)

    for score in list_fft_score_bad_reviews:
        list_combined_fft_scores.append(score)
    
    #randomly assign gender to test data
    random.seed(42)
    list_gender = [random.choice(['Male', 'Female']) for review in range(len(list_combined_reviews))]
    #randomly assign ethnicity to test data
    random.seed(42)
    list_ethnicity = [random.choice(['White', 'Mixed', 'Asian', 'Black', 'Other']) for review in range(len(list_combined_reviews))]
    #randomly assign a service label to test data
    random.seed(42)
    list_service = [random.choice(['Service A', 'Service B', 'Service C']) for review in range(len(list_combined_reviews))]

    data = [{
        "Review": list_combined_reviews[i], 
        "fft_score": list_combined_fft_scores[i],
        "gender": list_gender[i],
        "ethnicity": list_ethnicity[i],
        "service": list_service[i]} for i in range(len(list_combined_fft_scores))]
    
    df_fft_reviews = pd.DataFrame(data)

    #update the source data to have a single binary ground truth column
    string1 = 'very good'
    string2 = 'good'

    #original line for splitting into binary labels
    #df_fft_reviews['fft_binary'] = np.where((df_fft_reviews['fft_score'] == string1) | (df_fft_reviews['fft_score'] == string2), 'Positive', 'Negative')

    return df_fft_reviews
import os
import requests
import urllib.request
# from urllib.error import HTTPError
import json
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'architectural-style (jsuhleen@gmail.com)'
}

# this method accepts style as input, and outputs a list of subcategories from which we will pull further images
def subcategories(style):

    # initialize 
    sub_categories = []

    # all sub categories of archtectural styles, ordered
    if (style == 'Moorish architecture'):
        sub_categories = ['Islamic_architecture_in_Algeria', 'Moorish_merlons_in_Córdoba,_Spain', 'Royal_Chapel,_Mosque-Cathedral_of_Córdoba', 'Dar_Hassan_Pacha', 'Great_Mosque_of_Algiers', 'Almoravid_Koubba', 'Great_Mosque_of_Tlemcen']

    elif (style == 'Berber architecture'):
        sub_categories = ['Chaoui_architecture', 'Kabyle_architecture', 'Vernacular_architecture_of_Morocco', 'Ksar_Tamezmoute', 'Kasbahs_in_Morocco']

    elif (style == 'Hausa architecture'):
        sub_categories = ['Hausa_Traditional_Architecture', 'Hausa_architecture_in_Niger', 'Hausa_architecture_in_Nigeria']

    elif (style == 'Swahili architecture'):
        sub_categories = ['Kizimkazi_Mosque', 'Mosques_in_Dar_es_Salaam', 'Malindi_Mosque', "Sultan's_Palace,_Zanzibar", 'Aga_Khan_Mosque,_or_Ismaili_Jamatkhana', 'Mosques_in_Mombasa', 'Juma_mosque_(Malindi)', 'Rawdha_mosque_(Malindi)', 'Khoja_Mosque_(Nairobi)', 'Gongoni_Mosque', 'Sunni_Mosque_of_Arusha']

    elif (style == 'Maya architecture'):
        sub_categories = ['Puuc_architecture', 'Building_19_-_Yaxchilan', 'Palacio_del_Gobernador,_Uxmal', 'Tikal_Structure_5C-54_(Lost_World_Pyramid)', 'Tikal_Structure_5C-49_(Talud-Tablero_Temple)', 'Maya_pyramids_in_Quintana_Roo']
    
    elif (style == 'Pueblo architecture'):
        sub_categories = ['Acoma_Pueblo', 'Pueblo_de_Taos']

    elif (style == 'Brutalist architecture'):
        sub_categories = ['Brutalist_architecture_in_London', 'Brutalist_architecture_in_Australia', 'Brutalist_architecture_in_Argentina', 'Brutalist_architecture_in_Austria', 'Brutalist_architecture_in_Denmark', 'Brutalist_architecture_in_Italy', 'Brutalist_architecture_in_France', 'Brutalist_architecture_in_the_Netherlands']

    elif (style == 'Stalinist architecture'):
        sub_categories = ['300-seat_club_building_by_Konstantin_Bartoshevich', 'Club_type_088', 'Death_Tower_(administrative_building),_Perm', 'Stalinist_architecture_in_Tallinn', 'Stalinist_architecture_in_Minsk', 'Stalinist_architecture_in_Tajikistan', 'Stalinist_architecture_in_Kyrgyzstan', 'Stalinist_architecture_in_Latvia', 'Stalinist_architecture_in_Mongolia', 'Stalinist_architecture_in_Poland', 'Stalinist_architecture_in_Ukraine', 'Stalinist_architecture_in_Slovakia', 'Stalinist_architecture_in_Moldova', 'Stalinist_architecture_in_Kazakhstan']
    
    elif (style == 'Dravidian architecture'):
        sub_categories = ['Amritesvara_Temple,_Annigeri', 'Yellamma_Temple,_Badami', 'Kumbareshwar_Temple,_Lakkundi', 'Tarakesvara_Temple,_Hangal', 'Gopurams', 'Rajarajeshwari_Temple,_Bengaluru', 'Gopurams_in_Kerala']

    elif (style == 'Jain architecture in India'):
        sub_categories = ['Jain_temples_in_Delhi', 'Jain_temples_in_Karnataka', 'Jain_temples_in_Madhya_Pradesh', 'Badami_cave_temples_(cave_2)', 'Chitharal_Jain_Monuments']

    elif (style == 'Mughal architecture'):
        sub_categories = ['Daud_Khan_Fort', 'Roshanara_Bagh', 'Shah_Jahan_Mosque,_Woking', 'Tomb_of_Mariam_uz-Zamani', 'Hashtsal_Minar', 'Agra_Fort']

    elif (style == 'Balinese architecture'):
        sub_categories = ['Bali_Aga_architecture', 'Balinese_gates', 'Palaces_in_Bali', 'Kulkul_towers', 'Meru_towers', 'Wantilan']

    elif (style == 'Ottoman architecture'):
        sub_categories = ['Ottoman_houses_in_Gjirokastër', 'Ottoman_houses_in_Ankara', 'Ottoman_houses_in_Bursa', 'Ottoman_houses_in_Istanbul', 'Bazaraa_wikala', 'Ottoman_bird_palace', 'Ottoman_architecture_in_Albania', 'Ottoman_architecture_in_Bulgaria', 'Ottoman_architecture_in_Bosnia_and_Herzegovina', 'Ottoman_architecture_in_Kosovo', 'Ottoman_mosques_in_Greece', 'Ottoman_mosques_in_Albania', 'Ottoman_architecture_in_Turkey', 'Ottoman_architecture_in_Sarajevo', 'Mosque_of_Abu_Dahab', 'Ottoman_architecture_in_Cyprus']

    elif (style == 'Safavid architecture'):
        sub_categories = ['Safareh_House', 'Tiznoo_House', 'Adobe_Bridge_(Langarud)', 'Agha_Mahmud_Mosque_(Tehran)', 'Jameh_Mosque_of_Abarkuh', 'Ferdows_Madrasa', 'Rig_Mosque_(Yazd)', 'Juma_mosque_(Ganja)', 'Madrasas_of_Shah_Mosque_(Isfahan)', 'Gonabad_Madrasa', 'David_House_(Isfahan)', 'Ali_Qapu_(Qazvin)', 'Martha_Peters_House_(Isfahan)', 'Dehnamak_Caravanserai', 'Chaleh_Siah_Caravanserai', 'Golshan_Caravanserai_(Hamadan)', 'Hajib_Caravanserai', 'Tehran_Grand_Bazaar', 'Gonbad-e_Chahar_Suq_(Saveh)']

    elif (style == 'Khmer architecture'):
        sub_categories = ['Khmer_pagodas_in_Vietnam', 'Angkor_Wat_libraries', 'Angkor_Wat_main_temple', 'Face_towers', 'Oudong']

    elif (style == 'Architecture of Edo period'):
        sub_categories = ['Ichi-no-mon,_Matsuyama_Castle', 'Obama-Nishigumi', 'Kami-jinko,_Nikkō_Tōshō-gū', 'Kameyama_Gobo_Hontokuji', 'Kyu_Yagyuhan_karoyashiki', 'Nagaya_(apartment)', 'Nagaya-mon', 'Honbō,_Kongosanmaiin', 'Rokusho-jinja_(Okazaki)', "Shin'yo-sha,_Nikkō_Tōshō-gū", 'Hirokane_house,_Takahashi', 'Kagura-den,_Nikkō_Tōshō-gū']

    elif (style == 'Architecture of the Qing Dynasty'):
        sub_categories = ['Buildings_in_the_Summer_Palace,_Beijing', 'He_Garden', 'Mingzhi_Academy', 'Chongzheng_Hall', 'Dacheng_Hall_(Qufu_Confucian_Temple)', 'Sheqi_Shan-Shan_Guildhall', 'Puning_Temple', 'Lü_Zu_Temple,_Weihui', 'Palace_of_Earthly_Tranquility', 'Palace_of_Heavenly_Purity_(Forbidden_City)', 'Six_Eastern_Palaces', 'Yellow_Crane_Tower_(1868-1884)', 'House_of_the_Huangcheng_Chancellor']

    elif (style == 'Architecture of the Joseon Dynasty'):
        sub_categories = ['Hyanggyo', 'Seowon', 'Gaeksa', 'Jinnamgwan', 'Gangnyeongjeon', 'Daejojeon', 'Jagyeongjeon', 'Gyeonghuigung', 'Sujeongjeon', 'Hwaseomun', 'Irodang', 'Noandang', 'Bongmujeong', 'Bihyeongak', 'Jibokjae', 'Yeonghwadang', 'Jaseondang']

    elif (style == 'Buddhist architecture'):
        sub_categories = ['Buddhist_temples_in_Thailand', 'Bell_towers_in_Thailand', 'Stupas_in_Vietnam', 'Buddhist_monasteries', 'Buddhist_monasteries_in_Myanmar', 'Buddhist_temples_in_Myanmar']

    elif (style == 'Minangkabau architecture'):
        sub_categories = ['Minangkabau_houses', 'Minangkabau_mosques', 'Minangkabau_barns', 'Minangkabau_spires', 'Minangkabau_roofs']
    
    return sub_categories

def downloadImage(index, image, style):
            
    # define prefix
    download_prefix = "https://commons.wikimedia.org/wiki/"
    # then url 
    url = download_prefix + image

    # get response 
    response = requests.get(url)
    # make it into soup
    soup = BeautifulSoup(response.content, 'html.parser')
    # draw the imaage by looking for the 'alt' key which should say image
    actual_image = soup.find('img', {'alt': image})
    # get the name of the image to download it
    actual_image_name = str(index) + ".jpg"
    # determine path to which image should be downloaded 
    download_path = os.path.join("data", style, actual_image_name)

    # try downloading
    try:
        # image url is in the 'src' key
        urllib.request.urlretrieve(actual_image['src'], download_path)  
    
    # if not
    except Exception as e:
        # skip file
        print("Unable to download file. Skipping...")  
        # need to return false for later purposes 
        return False
    
    return True


# this method accepts the index of an image, the actual json object, and the name of the style 
# it will parse the json to store the attributes to file data/style/attributes.txt
def storeAttributes(index, image, style):

    # define the image prefix and suffix urls 
    image_prefix = 'https://en.wikipedia.org/w/api.php?action=query&prop=imageinfo&iiprop=extmetadata&titles='
    image_suffix = '&format=json'

    # determine current image's url 
    current_image_url = image_prefix + image + image_suffix

    # get response 
    image_response = requests.get(current_image_url, headers=headers)
    image_response = json.loads(image_response.text)

    # retrieve list of objects 
    halfway_metadata = image_response['query']['pages']

    # get the key 
    for key in halfway_metadata:
        # like, huh? what do you mean the key is suddenly 2374u9347
        weird_key = key

    # finish metadata
    try:
        metadata = halfway_metadata[weird_key]['imageinfo'][0]['extmetadata']
    # if even this doesn't work, just give up
    except Exception as e:
        return

    # define path to attributes 
    attributes_path = os.path.join("data", style, "attributes.txt")
    # open attributes txt
    attributes_txt = open(attributes_path, "a", encoding="utf-8")

    # write index
    attributes_txt.write(str(index) + ".\n")
    # title 
    title = metadata['ObjectName']['value']
    # write to txt
    attributes_txt.write("  Title: {ftitle}\n".format(ftitle=title))

    # creator html
    try:
        # sometimes they don't have a name
        creator_html = metadata['Artist']['value']
        # use bs4 to extract creator
        creator = BeautifulSoup(creator_html, 'html.parser').get_text()
    # in which case designate unknown
    except Exception as e:
        creator = "Unknown"
    # write to txt
    attributes_txt.write("  Creator: {fcreator}\n".format(fcreator=creator))

    # original link
    original_link = "https://commons.wikimedia.org/wiki/" + image # huh?
    # write to txt
    attributes_txt.write("  Source: Wikimedia, {flink}\n".format(flink=original_link))

    # license
    try:
        license = metadata['UsageTerms']['value']
    except KeyError:
        # sometimes they don't list licenses properly, then just use this
        license = metadata['LicenseShortName']['value']
    # write to txt
    attributes_txt.write("  Copyright information: {flicense}\n\n".format(flicense=license))

    # close attributes txt
    attributes_txt.close()

if __name__ == '__main__':
    
    style_list = ['Moorish architecture', 'Berber architecture', 'Hausa architecture', 'Swahili architecture', 'Maya architecture', 'Pueblo architecture', 'Brutalist architecture', 'Stalinist architecture', 'Minangkabau architecture', 'Dravidian architecture', 'Jain architecture in India', 'Mughal architecture', 'Balinese architecture', 'Ottoman architecture', 'Safavid architecture', 'Khmer architecture', 'Architecture of Edo Period', 'Architecture of the Qing Dynasty', 'Architecture of the Joseon Dynasty', 'Buddhist architecture']

    # go through each style
    for style in style_list: 

        # retrieve sub-categories
        sub_categories = subcategories(style)
        # current path with data and style 
        current_path = os.path.join('data', style)

        # make a directory for this style. 
        try: 
            os.mkdir(current_path)
        # handle error
        except Exception as e: 
            # continue that 
            print("Style folder exists. Continuing...")

        # designate url pre/suffix
        search_prefix = 'https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:'
        search_suffix = '&cmlimit=500&cmtype=file&format=json'

        # equivalent of searching images for this style. combine image search url with the search query
        url = search_prefix + style + search_suffix

        # get response
        response = requests.get(url, headers=headers)
        response = json.loads(response.text)

        # designate parent dictionary - the main styles page 
        parent_dictionary = response['query']['categorymembers']
        # store length of parent image list
        accumulated_len = len(parent_dictionary)

        # go through this style page
        for i, parent_object in enumerate(parent_dictionary):
            # get the image title 
            parent_image = parent_object['title']
            # download image
            download_attempt = downloadImage(i+1, parent_image, style)
            # make path to attributes txt
            if (download_attempt == False):
                continue
            # store to attributes txt
            storeAttributes(i+1, parent_image, style)

        # now go through the sub categories
        for i, sub_category in enumerate(sub_categories):
            # get the list of images
            url = search_prefix + sub_category + search_suffix

            # get response
            response = requests.get(url, headers=headers)
            response = json.loads(response.text)

            # retrieve the sub dictionary
            sub_dictionary = response['query']['categorymembers']

            # then go through that 
            for j, sub_object in enumerate(sub_dictionary):
                # retrieve title 
                sub_image = sub_object['title']
                # download image
                download_attempt = downloadImage(accumulated_len+j+1, sub_image, style)
                # if attempt fails, continue
                if (download_attempt == False):
                    continue
                # store to attributes txt 
                storeAttributes(accumulated_len+j+1, sub_image, style)
            
            # this is to keep track of the index so that we can label the images 
            accumulated_len += len(sub_dictionary)

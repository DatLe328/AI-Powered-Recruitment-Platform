with open('data/cleaned_skills_3.txt', 'r', encoding='utf-8') as infile, \
     open('cleaned_skills_3_out.txt', 'w', encoding='utf-8') as outfile:
    for line in infile:
        cleaned_line = line.strip().strip('"')
        outfile.write(cleaned_line + '\n')
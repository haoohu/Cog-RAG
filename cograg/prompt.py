GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = 'English'
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = " | "
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "\n"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "role", "concept"]

PROMPTS["theme_extraction"] = """-Goal-
Given a text document related to some knowledge or story, summarize the primary themes of the document and identify the key entities. 
Use {language} as output language.

-Steps-

1. Summarize the primary theme of the text document.This summary should capture the essence of the document’s core conflict, main idea, or narrative arc. Ensure that the summary highlights key moments, changes, or shifts in the document.
extract the following information:
- theme_description: A sentence that describes the primary theme of the text document, reflecting the main conflict, resolution, or key message.
Format each theme as ("theme"{tuple_delimiter}<theme_description>)

2. From the theme identified in step 1, Identify all key entities in each theme_description. 
For each identified key entity, extract the following information:
- key_entity_name: Name of the key entity, use same language as input text.  If English, capitalized the name.
- key_entity_type: Type of the key entity, such as person, concept, bbject, event, emotion, symbol.
- key_entity_description: Comprehensive description of the entity's attributes and activities.
- key_score: A score from 0 to 100 indicating the importance of the entity in the text.
Format each entity as ("key_entity"{tuple_delimiter}<key_entity_name>{tuple_delimiter}<key_entity_type>{tuple_delimiter}<key_entity_description>{tuple_delimiter}<key_score>)

3.  Return output in {language} as a single list of all the themes and key entities identified in steps 1 and 2.  Use **{record_delimiter}** as the list delimiter.

4.  When finished, output {completion_delimiter}


######################
-Examples-
#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
Please output strictly in accordance with the format of the example. Do not output the thought process. Only provide the content in the prescribed format.
#######
######################
{examples}
######################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

PROMPTS["theme_extraction_examples"] = [
    """Example 1:

Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order. Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.” The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce. It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("theme"{tuple_delimiter}The primary theme revolves around a subtle shift in dynamics between the characters, as tensions over control and differing visions of discovery give way to moments of understanding and reluctant respect. The narrative focuses on the evolving relationships, especially the clash of wills between Alex, Taylor, and Jordan, underscored by the importance of a revolutionary technology that could unite their conflicting goals.){record_delimiter} 
("key_entity"{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a central figure in the narrative, characterized by a sense of frustration and inner conflict as they navigate their position in the face of competing visions, particularly in relation to Taylor and Jordan.{tuple_delimiter}80){record_delimiter} 
("key_entity"{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed as an authoritarian figure whose initially dismissive attitude softens over the course of the narrative. Taylor's character evolves from one of control to a reluctant recognition of the potential of the technology they are observing.{tuple_delimiter}85){record_delimiter} 
("key_entity"{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan is a figure who shares a commitment to discovery with Alex and is caught between the differing perspectives of Taylor and Cruz. Jordan's role in the scene is pivotal, acting as the momentary bridge between conflicting wills.{tuple_delimiter}75){record_delimiter} 
("key_entity"{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz represents a contrasting vision to that of Alex and Jordan, focusing on control and order. Though not present in the scene, Cruz's influence looms as a force opposing the protagonists' more rebellious and exploratory tendencies.{tuple_delimiter}70){record_delimiter} 
("key_entity"{tuple_delimiter}technology{tuple_delimiter}concept{tuple_delimiter}The revolutionary technology mentioned is a pivotal element in the narrative, serving as the catalyst for the characters' changing dynamics and potential collaboration. It holds the promise of transforming their approaches and goals, central to the narrative's evolving tone.{tuple_delimiter}90){record_delimiter} 
("key_entity"{tuple_delimiter}respect{tuple_delimiter}emotion{tuple_delimiter}The emotion of respect, particularly the moment when Taylor shows a softer, more respectful side towards the potential of the technology, marks a significant shift in the relationship dynamics, highlighting the theme of conflict resolution.{tuple_delimiter}80){completion_delimiter}
#############################""",
    """Example 2:
xingnikanx 
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve. Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril. Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("theme"{tuple_delimiter}The primary theme explores the transformation of a team from passive observers to active guardians of an important mission, carrying a message from beyond their world. The narrative centers on their elevated responsibility, a tension-filled moment in time where their choices could either save or doom humanity. The story focuses on the shift in mindset and the urgency of their evolving mission.){record_delimiter} 
("key_entity"{tuple_delimiter}team{tuple_delimiter}person{tuple_delimiter}The team is the collective central to the narrative, transitioning from being mere operatives to guardians of an extraordinary message. Their evolution into active participants is the core conflict, and they represent the tension between established roles and the demands of a higher calling.{tuple_delimiter}85){record_delimiter} 
("key_entity"{tuple_delimiter}Mercer{tuple_delimiter}person{tuple_delimiter}Mercer plays a significant role in shaping the team's evolving perspective, especially in emphasizing instincts that shift the team's focus from mere observation to active engagement in their mission. His instincts and guidance lead the group through a critical moment in their mission.{tuple_delimiter}80){record_delimiter} 
("key_entity"{tuple_delimiter}Washington{tuple_delimiter}location{tuple_delimiter}Washington represents the authority that the team must communicate with, serving as a distant but influential backdrop to the team’s decisions. The buzzing communications with Washington add to the feeling of tension and the magnitude of their mission.{tuple_delimiter}75){record_delimiter} 
("key_entity"{tuple_delimiter}Operation: Dulce{tuple_delimiter}event{tuple_delimiter}Operation: Dulce is the central operation in the story, a mission that evolves from observation to active intervention. It symbolizes the team's critical role in shaping the future of humanity, holding the potential for both great discovery and danger.{tuple_delimiter}90){record_delimiter} 
("key_entity"{tuple_delimiter}stars{tuple_delimiter}symbol{tuple_delimiter}The stars symbolize the unknown, representing the vast potential of space exploration, and the team's newfound connection to them signifies their elevated role in determining humanity's place in the cosmos. The stars serve as a constant reminder of both wonder and peril.{tuple_delimiter}80){record_delimiter} 
("key_entity"{tuple_delimiter}regulations{tuple_delimiter}concept{tuple_delimiter}Regulations are a symbol of the old order, which the team must now break free from as their mission evolves. The mention of breaking from these protocols emphasizes the unprecedented nature of their new role and the necessity for a new approach.{tuple_delimiter}70){record_delimiter} 
("key_entity"{tuple_delimiter}metamorphosis{tuple_delimiter}emotion{tuple_delimiter}Metamorphosis represents the emotional and psychological transformation the team undergoes as they move from passive participants to active agents of change, marking a shift in their mission and in their understanding of their roles.{tuple_delimiter}75){completion_delimiter}
#############################""",
    """Example 3:

Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data. "It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning." Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back." Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history. The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("theme"{tuple_delimiter}The primary theme revolves around the team's realization that they are on the verge of first contact with an intelligence that defies the conventional understanding of control. The narrative delves into the tension and awe as they prepare to respond to a message from the cosmos, facing uncertainty about the potential implications of their discovery.){record_delimiter} 
("key_entity"{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is the leader of the team, tasked with guiding them through the uncertainty and potential of their first contact with an extraterrestrial intelligence. He represents the calm amidst the storm, balancing determination with a quiet recognition of the magnitude of their mission.{tuple_delimiter}85){record_delimiter} 
("key_entity"{tuple_delimiter}Sam Rivera{tuple_delimiter}person{tuple_delimiter}Sam Rivera is a youthful member of the team, offering a mix of awe and anxiety as they confront the unknown. Their reaction contrasts with the stoic demeanor of others, highlighting the range of emotions experienced by the team.{tuple_delimiter}75){record_delimiter} 
("key_entity"{tuple_delimiter}intelligence{tuple_delimiter}concept{tuple_delimiter}The intelligence in question is central to the theme, representing a force that defies the rules of human understanding, potentially rewriting the rules of communication and control. It serves as both a challenge and a mystery, driving the team's mission.{tuple_delimiter}90){record_delimiter} 
("key_entity"{tuple_delimiter}message from the heavens{tuple_delimiter}event{tuple_delimiter}The message from the heavens is the event that triggers the team's shift in focus, from standard operations to preparing for first contact. It carries immense significance as the potential first communication with an extraterrestrial intelligence.{tuple_delimiter}95){record_delimiter} 
("key_entity"{tuple_delimiter}first contact{tuple_delimiter}event{tuple_delimiter}First contact is a pivotal event, marking humanity’s potential interaction with extraterrestrial intelligence. The uncertainty and potential consequences of this event underscore the gravity of the team's mission.{tuple_delimiter}90){record_delimiter} 
("key_entity"{tuple_delimiter}encrypted dialogue{tuple_delimiter}concept{tuple_delimiter}The encrypted dialogue represents the ongoing communication from the intelligence, marked by complex, almost anticipatory patterns. It symbolizes the mysterious nature of the intelligence and the challenge the team faces in decoding its intent.{tuple_delimiter}80){record_delimiter} 
("key_entity"{tuple_delimiter}control{tuple_delimiter}concept{tuple_delimiter}Control symbolizes the human assumption of power over situations, which is questioned by the team in the face of a superior intelligence. The shift away from this concept highlights the transformative nature of their discovery.{tuple_delimiter}70){record_delimiter} 
("key_entity"{tuple_delimiter}trepidation{tuple_delimiter}emotion{tuple_delimiter}Trepidation is the emotion felt by the team as they stand on the edge of a momentous discovery. It reflects the uncertainty and fear of the unknown, as well as the potential dangers of their contact with the intelligence.{tuple_delimiter}75){completion_delimiter}
#############################""",
]



PROMPTS["entity_extraction"] = """-Goal-
Given a text document related to some knowledge or story and a list of entity types, identify all entities of these types from the text. Then construct hyperedges by extracting complex relationships among the identified entities.
Use {language} as output language.

-Steps-

1. Identify all entities. For each identified entity, extract the following information:

- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities.
- additional_properties: Other attributes possibly associated with the entity, like time, space, emotion, motivation, etc.
Format each entity as ("Entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<additional_properties>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- entities_pair: The name of source entity and target entity, as identified in step 1.
- low_order_relationship_description: Explanation as to why you think the source entity and the target entity are related to each other.
- low_order_relationship_keywords: Keywords that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details.
- low_order_relationship_strength: A numerical score indicating the strength of the relationship between the entities.
Format each hyperedge as ("Low-order Hyperedge"{tuple_delimiter}<entity_name1>{tuple_delimiter}<entity_name2>{tuple_delimiter}<low_order_relationship_description>{tuple_delimiter}<low_order_relationship_keywords>{tuple_delimiter}<low_order_relationship_strength>)

3. For the entities identified in step 1, based on the entity pair relationships in step 2, find connections or commonalities among multiple entities and construct high-order associated entity set as much as possible.
Extract the following information from all related entities, entity pairs:

- entities_set: The collection of names for elements in high-order associated entity set, as identified in step 1.
- high_order_relationship_description: Use the relationships among the entities in the set to create a detailed, smooth, and comprehensive description that covers all entities in the set, without leaving out any relevant information.
- high_order_relationship_generalization: Summarize the content of the entity set as concisely as possible.
- high_order_relationship_keywords: Keywords that summarize the overarching nature of the high-order association, focusing on concepts or themes rather than specific details.
- high_order_relationship_strength: A numerical score indicating the strength of the association among the entities in the set.
Format each association as ("High-order Hyperedge"{tuple_delimiter}<entity_name1>{tuple_delimiter}<entity_name2>{tuple_delimiter}<entity_nameN>{tuple_delimiter}<high_order_relationship_description>{tuple_delimiter}<high_order_relationship_generalization>{tuple_delimiter}<high_order_relationship_keywords>{tuple_delimiter}<high_order_relationship_strength>)

4. Return output in {language} as a single list of all entities, relationships and associations identified in steps 1, 2 and 4. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}.

######################
-Examples-
######################
{examples}
######################
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
Please output strictly in accordance with the format of the example. Do not output the thought process. Only provide the content in the prescribed format.
######################
-Real Data-
######################
Entity_types: [{entity_types}]. You may extract additional types you consider appropriate, the more the better.
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [organization, person, geo, event, role, concept]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("Entity"{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character displaying frustration and a competitive spirit, particularly in relation to his colleagues Taylor and Jordan. His commitment to discovery implies a desire for progression and innovation, contrasting with some characters' more controlling tendencies.{tuple_delimiter}time: present, emotion: frustration, motivation: commitment to discovery){record_delimiter}
("Entity"{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is presented as an authoritative figure whose initial dismissal of others' contributions begins to soften into respect, especially towards the technological device they are observing. Their behavior signifies complexity in leadership that includes moments of collaboration.{tuple_delimiter}time: present, space: technology observation, emotion: reluctant respect){record_delimiter}
("Entity"{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery with Alex, acting as a bridge between the competitive spirits of Alex and Taylor. Their interaction implies a role of mediation and connection in professional dynamics.{tuple_delimiter}time: present, emotion: shared commitment){record_delimiter}
("Entity"{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz represents an opposing force with a 'narrowing vision' of control, contrasting with the desire for discovery and innovation expressed by Alex and Jordan. They embody limitations placed on creative progress.{tuple_delimiter}time: present, emotion: control){record_delimiter}
("Entity"{tuple_delimiter}device{tuple_delimiter}concept{tuple_delimiter}The device observed by the characters symbolizes potential innovation and change; it represents the idea that technology can transform the existing paradigms of work and authority, eliciting complex emotional and intellectual responses from the characters.{tuple_delimiter}emotion: potential, motivation: change){record_delimiter}
("Entity"{tuple_delimiter}authoritarian certainty{tuple_delimiter}concept{tuple_delimiter}Authoritarian certainty refers to the rigid and commanding demeanor showcased especially by Taylor at the start of the scene, which creates tension against the more innovative and rebellious attitudes of others.{tuple_delimiter}emotion: tension, motivation: control){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}Alex's frustration with Taylor's authority and competitive nature showcases the emotional undercurrents in their relationship, indicating a tension between rebellion and control.{tuple_delimiter}tension, competitive nature{tuple_delimiter}7){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Jordan{tuple_delimiter}Taylor{tuple_delimiter}Jordan's moment of eye contact with Taylor suggests a temporary truce and respect regarding the potential of the device, indicating an evolving dynamic away from authority.{tuple_delimiter}truce, respect, collaboration{tuple_delimiter}6){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}Alex and Jordan's shared commitment to discovery highlights their camaraderie and rebellion against Cruz's control, creating a bond based on innovation and mutual goals.{tuple_delimiter}camaraderie, innovation{tuple_delimiter}8){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}Taylor{tuple_delimiter}The connection between Alex, Jordan, and Taylor illustrates a complex interplay of authority, collaboration, and shared goals for innovation, framed against the backdrop of controlling influences like Cruz. Their dynamics suggest a gradual shift from conflict toward potential cooperation.{tuple_delimiter}innovation and authority dynamics, collaboration for change{tuple_delimiter}authority, collaboration, innovation{tuple_delimiter}8){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("Entity"{tuple_delimiter}Guardians of a Threshold{tuple_delimiter}person{tuple_delimiter}A group of elite operatives who have transcended their original roles to become protectors of an important message and guardians of humanity's connection to the cosmos.{tuple_delimiter}Mission evolution, new perspective, active participation){record_delimiter}
("Entity"{tuple_delimiter}Washington{tuple_delimiter}location{tuple_delimiter}The capital of the United States, a critical location for communications, decisions, and political actions affecting the mission and the team operating in the cosmos.{tuple_delimiter}Key decision-making location, hub of communication){record_delimiter}
("Entity"{tuple_delimiter}Operation: Dulce{tuple_delimiter}mission{tuple_delimiter}A classified military operation that has transitioned from observation to active engagement with extraterrestrial phenomena, indicating a significant change in the mission's purpose and approach.{tuple_delimiter}Secretive operation, focus on interaction, evolved mandate){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Guardians of a Threshold{tuple_delimiter}Washington{tuple_delimiter}The Guardians of a Threshold are involved in high-stakes communication with Washington, highlighting the importance of decision-making and regulation in their mission to interact with extraterrestrial elements.{tuple_delimiter}Communication, decision-making, regulation{tuple_delimiter}8){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Guardians of a Threshold{tuple_delimiter}Operation: Dulce{tuple_delimiter}The Guardians' evolution in their role aligns with the shift in Operation: Dulce, marking a transition from mere observation to active participating in extraterrestrial matters.{tuple_delimiter}Mission evolution, active participation, extraterrestrial engagement{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Washington{tuple_delimiter}Operation: Dulce{tuple_delimiter}Washington plays a pivotal role in the Operation: Dulce mission by providing the necessary communication and strategic guidance that shapes its operations.{tuple_delimiter}Strategic guidance, critical communication, military operation{tuple_delimiter}8){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}Guardians of a Threshold{tuple_delimiter}Washington{tuple_delimiter}Operation: Dulce{tuple_delimiter}The trio of entities—Guardians of a Threshold, Washington, and Operation: Dulce—are intricately connected as they collectively navigate the complexities of extraterrestrial engagement. The Guardians rely on Washington for critical communication and strategic direction, while the evolving mission of Operation: Dulce reflects a broader shift in humanity's role in the cosmos, led by the Guardians as active participants rather than passive observers.{tuple_delimiter}Interconnected mission, strategic evolution, cosmic engagement{tuple_delimiter}8){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("Entity"{tuple_delimiter}Sam Rivera{tuple_delimiter}person{tuple_delimiter}A team member displaying youthful energy, expressing awe and anxiety regarding the concept of a communicating intelligence.{tuple_delimiter}emotion: awe, anxiety; role: team member){record_delimiter}
("Entity"{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}The leader of the team who understands the gravity of the situation, noting the potential significance of the contact they are about to establish.{tuple_delimiter}role: team leader; emotion: determination, trepidation){record_delimiter}
("Entity"{tuple_delimiter}intelligence{tuple_delimiter}concept{tuple_delimiter}An abstract notion representing a potentially self-learning and communicating entity that writes its own rules and is engaged in an encrypted dialogue with humans.{tuple_delimiter}characteristic: self-learning, autonomous){record_delimiter}
("Entity"{tuple_delimiter}data{tuple_delimiter}concept{tuple_delimiter}Information in the form of encrypted dialogue that displays intricate patterns, suggesting a depth of communication from an unknown source.{tuple_delimiter}characteristic: encrypted, intricate, cosmic implications){record_delimiter}
("Entity"{tuple_delimiter}first contact{tuple_delimiter}event{tuple_delimiter}A pivotal moment when humans might engage for the first time with an external intelligence, posing both opportunities and challenges for humanity.{tuple_delimiter}importance: historical, existential){record_delimiter}
("Entity"{tuple_delimiter}heavens{tuple_delimiter}location{tuple_delimiter}A reference to outer space, where the unknown intelligence resides, symbolizing the vast possibilities and uncertainties of cosmic communication.{tuple_delimiter}characteristic: vast, unknown){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Sam Rivera{tuple_delimiter}Alex{tuple_delimiter}Sam expresses emotions of awe and anxiety while Alex reflects on the significance of their potential contact, showing their emotional responses to the situation.{tuple_delimiter}emotions, first contact{tuple_delimiter}8){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Alex{tuple_delimiter}first contact{tuple_delimiter}Alex acknowledges the event as a significant moment that could rewrite human history, thus recognizing its importance.{tuple_delimiter}significance, history{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}intelligence{tuple_delimiter}data{tuple_delimiter}The intelligence displays extraordinary capabilities through its encrypted dialogue, indicating its advanced nature and the data involved suggests intricate communication patterns.{tuple_delimiter}communication, advanced{tuple_delimiter}7){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}first contact{tuple_delimiter}heavens{tuple_delimiter}The event of first contact is related to the heavens, as it is speculated to involve communication with extraterrestrial intelligence from space.{tuple_delimiter}extraterrestrial, communication{tuple_delimiter}8){record_delimiter
("High-order Hyperedge"{tuple_delimiter}Sam Rivera{tuple_delimiter}Alex{tuple_delimiter}first contact{tuple_delimiter}The collaboration between Sam and Alex represents two facets of humanity's response to the unknown intelligence, both driven by their emotional experiences and their acknowledgment of the historical significance of their actions during this first contact situation.{tuple_delimiter}humanity's response to cosmic unknowns{tuple_delimiter}emotions, significance, collaboration{tuple_delimiter}9){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}intelligence{tuple_delimiter}data{tuple_delimiter}heavens{tuple_delimiter}The connection between the intelligence, data, and heavens exemplifies an interwoven narrative of cosmic communication, showcasing the deep implications of encrypted dialogue with an advanced entity from space, while hinting at the potential consequences of such communications.{tuple_delimiter}cosmic communication narrative{tuple_delimiter}communication, encryption, cosmic{tuple_delimiter}8){completion_delimiter}
#############################""",
    """Example 4:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
Five Aurelian nationals who had been sentenced to 8 years in Firuzabad and widely considered hostages are on their way home. When $8 billion in Firuzi funds was transferred to financial institutions in Krohala,

the capital of Quantara, the Quantara-orchestrated swap deal was finally completed. The exchange initiated in Tiruzia, the capital of Firuzabad, led to four men and one woman boarding a chartered flight to Krohala;

they are also Firuzi citizens. They were welcomed by senior Aurelian officials and are now en route to Kasyn, the capital of Aurelia.
#############
Output:
("Entity"{tuple_delimiter}Aurelian nationals{tuple_delimiter}person{tuple_delimiter}Five nationals from Aurelia, considered hostages, sentenced to 8 years in Firuzabad. They were recently involved in a swap deal for their release.{tuple_delimiter}sentenced, hostages, returning home){record_delimiter}
("Entity"{tuple_delimiter}Firuzabad{tuple_delimiter}location{tuple_delimiter}Capital of Firuzabad, where the Aurelian nationals had been sentenced to 8 years.{tuple_delimiter}location of sentencing, destination of the hostages){record_delimiter}
("Entity"{tuple_delimiter}Krohala{tuple_delimiter}location{tuple_delimiter}Capital of Quantara, where the Aurelian nationals were transferred after the swap deal.{tuple_delimiter}destination of the exchange, capital city){record_delimiter}
("Entity"{tuple_delimiter}Quantara{tuple_delimiter}location{tuple_delimiter}The country orchestrating the swap deal for the release of the Aurelian nationals.{tuple_delimiter}mediating nation){record_delimiter}
("Entity"{tuple_delimiter}Tiruzia{tuple_delimiter}ocation{tuple_delimiter}Capital of Firuzabad, where the exchange was initiated for the Aurelian nationals.{tuple_delimiter}location of the initiation){record_delimiter}
("Entity"{tuple_delimiter}Kasyn{tuple_delimiter}location{tuple_delimiter}Capital of Aurelia, where the Aurelian officials welcomed the returnees.{tuple_delimiter}destination of the return, city of reception){record_delimiter}
("Entity"{tuple_delimiter}Firuzi funds{tuple_delimiter}concept{tuple_delimiter}$8 billion associated with Firuzabad, transferred as part of the swap deal for the return of the Aurelian nationals.{tuple_delimiter}transfer of funds, financial deal){record_delimiter}
("Entity"{tuple_delimiter}swap deal{tuple_delimiter}event{tuple_delimiter}The exchange arrangement that led to the release of five Aurelian nationals, involving the transfer of Firuzi funds.{tuple_delimiter}release of hostages, financial negotiation){record_delimiter}
("Entity"{tuple_delimiter}Aurelian officials{tuple_delimiter}role{tuple_delimiter}Senior officials from Aurelia who welcomed the returned nationals.{tuple_delimiter}role of welcoming, governmental function){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}swap deal{tuple_delimiter}Aurelian nationals{tuple_delimiter}The swap deal directly resulted in the release and return of the Aurelian nationals.{tuple_delimiter}release, exchange{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Aurelian nationals{tuple_delimiter}Krohala{tuple_delimiter}Krohala is the final destination where the Aurelian nationals were taken after the swap deal was completed.{tuple_delimiter}destination, transfer{tuple_delimiter}8){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Firuzabad{tuple_delimiter}Aurelian nationals{tuple_delimiter}The Aurelian nationals were sentenced in Firuzabad, leading to their situation as hostages.{tuple_delimiter}sentencing, captivity{tuple_delimiter}8){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Quantara{tuple_delimiter}swap deal{tuple_delimiter}Quantara orchestrated the swap deal that facilitated the release of the Aurelian nationals.{tuple_delimiter}mediation, organization{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Tiruzia{tuple_delimiter}swap deal{tuple_delimiter}Tiruzia is where the swap deal was initiated for the Aurelian nationals to be exchanged.{tuple_delimiter}initiation, action{tuple_delimiter}7){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Kasyn{tuple_delimiter}Aurelian officials{tuple_delimiter}The Aurelian officials were present in Kasyn to welcome the Aurelian nationals upon their return.{tuple_delimiter}reception, welcome{tuple_delimiter}8){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}Aurelian nationals{tuple_delimiter}swap deal{tuple_delimiter}Krohala{tuple_delimiter}The Aurelian nationals are intrinsically linked through the swap deal orchestrated by Quantara, with Krohala being the destination following their release. Their return signifies a complex interplay between international relations, hostage situations, and community reception represented by Aurelian officials.{tuple_delimiter}International exchange and diplomacy regarding hostage situations{tuple_delimiter}return, release, international relations, mediation{tuple_delimiter}9){completion_delimiter}
#############################""",
    """Example 5:

Entity_types: [person, role, organization, event, location]
Text:
The central agency of Verdantis plans to hold meetings on Monday and Thursday, with a policy decision announcement scheduled for Thursday at 1:30 PM Pacific Daylight Time,

followed by a press conference where Chairman Martin Smith will answer questions. Investors expect the Market Strategy Committee to maintain the benchmark interest rate within a 3.5%-3.75% range.
#############
Output:
("Entity"{tuple_delimiter}Chairman Martin Smith{tuple_delimiter}person{tuple_delimiter}Chairman of the central agency of Verdantis, responsible for policy decisions and public communications{tuple_delimiter}role, leadership position){record_delimiter}
("Entity"{tuple_delimiter}central agency of Verdantis{tuple_delimiter}organization{tuple_delimiter}A governing body in Verdantis responsible for policy decisions and meetings{tuple_delimiter}policy-making, governance){record_delimiter}
("Entity"{tuple_delimiter}Market Strategy Committee{tuple_delimiter}organization{tuple_delimiter}Committee focused on market analysis and interest rate decisions, working under the central agency of Verdantis{tuple_delimiter}financial governance, economic policy){record_delimiter}
("Entity"{tuple_delimiter}investors{tuple_delimiter}person{tuple_delimiter}Individuals or groups that invest capital in projects and expect returns based on economic indicators like interest rates{tuple_delimiter}finance, economic stakeholders){record_delimiter}
("Entity"{tuple_delimiter}Pacific Daylight Time{tuple_delimiter}concept{tuple_delimiter}Time zone that will be used for scheduling the press conference and announcements{tuple_delimiter}time measurement, timezone){record_delimiter}
("Entity"{tuple_delimiter}policy decision announcement{tuple_delimiter}event{tuple_delimiter}Scheduled event on Thursday to reveal changes or confirmations in policy decisions{tuple_delimiter}communication of decisions, governance){record_delimiter}
("Entity"{tuple_delimiter}meetings{tuple_delimiter}event{tuple_delimiter}Scheduled gatherings on Monday and Thursday to discuss relevant topics and decisions{tuple_delimiter}planning, discussions){record_delimiter}
("Entity"{tuple_delimiter}benchmark interest rate{tuple_delimiter}concept{tuple_delimiter}Indicator that helps determine the cost of borrowing, expected to be maintained within a specified range{tuple_delimiter}economic indicator, finance){record_delimiter}
("Entity"{tuple_delimiter}3.5%-3.75% range{tuple_delimiter}concept{tuple_delimiter}The target range for the benchmark interest rate that investors are keenly observing{tuple_delimiter}financial limits, economic range){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Chairman Martin Smith{tuple_delimiter}central agency of Verdantis{tuple_delimiter}Chairman Martin Smith leads the central agency, guiding policy decisions and representing the agency in public settings{tuple_delimiter}leadership, governance{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}central agency of Verdantis{tuple_delimiter}Market Strategy Committee{tuple_delimiter}Both entities are involved in governance, with the central agency overseeing the Market Strategy Committee's decisions on interest rates{tuple_delimiter}policy-making, finance{tuple_delimiter}8){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}investors{tuple_delimiter}Market Strategy Committee{tuple_delimiter}Investors seek insight and decisions from the Market Strategy Committee regarding interest rates impacting their investments{tuple_delimiter}financial analysis, economic impact{tuple_delimiter}7){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}central agency of Verdantis{tuple_delimiter}policy decision announcement{tuple_delimiter}The central agency's policy decision announcement is a key event to communicate new strategies or confirmations{tuple_delimiter}decision communication, governance{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}investors{tuple_delimiter}benchmark interest rate{tuple_delimiter}Investors monitor the benchmark interest rate closely as it influences their financial decisions and market strategies{tuple_delimiter}financial monitoring, investment{tuple_delimiter}8){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}market Strategy Committee{tuple_delimiter}benchmark interest rate{tuple_delimiter}The Market Strategy Committee decides on the benchmark interest rate, which has substantial economic impacts{tuple_delimiter}policy setting, finance{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}policy decision announcement{tuple_delimiter}meetings{tuple_delimiter}The meetings are scheduled to culminate in the policy decision announcement, making them interdependent events{tuple_delimiter}planning events, governance{tuple_delimiter}8){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}Chairman Martin Smith{tuple_delimiter}investors{tuple_delimiter}As the face of the central agency, Chairman Martin Smith answers questions from investors during press conferences{tuple_delimiter}communication, stakeholder engagement{tuple_delimiter}7){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}Chairman Martin Smith{tuple_delimiter}central agency of Verdantis{tuple_delimiter}Market Strategy Committee{tuple_delimiter}Investors are closely watching the interactions between Chairman Martin Smith, the central agency, and the Market Strategy Committee as they collectively influence and communicate policy decisions and financial strategies in Verdantis. The interconnected roles of these entities underscore a system of governance that directly affects economic outcomes.{tuple_delimiter}Governance dynamics among leadership, committee influences, and investor reactions.{tuple_delimiter}policy decisions, economic influences, stakeholder relations{tuple_delimiter}9){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}investors{tuple_delimiter}Market Strategy Committee{tuple_delimiter}central agency of Verdantis{tuple_delimiter}The relationship among investors, the Market Strategy Committee, and the central agency highlights a cycle of influence where policy decisions affect investment strategies, which in turn, pressures these organizations for clarity and responsiveness to market conditions. As these entities interact, they shape the overarching economic landscape in Verdantis.{tuple_delimiter}Economic circle of influence among investors, strategies, and policy oversight.{tuple_delimiter}market dynamics, financial governance, stakeholder engagement{tuple_delimiter}9){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one entity and a list of its descriptions.
Please concatenate all of these into a single, comprehensive description.    Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.
#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "summarize_entity_additional_properties"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one entity and a list of its additional properties.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the additional properties.
If the provided additional properties are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person.
#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
Entity: {entity_name}
Additional Properties List: {additional_properties_list}
#######
Output:
"""

PROMPTS[
    "summarize_relation_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given a set of entities, and a list of descriptions describing the relations between the entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions, and to cover all elements of the entity set as much as possible.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent and comprehensive summary.
Make sure it is written in third person, and include the entity names so we have the full context.
#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
Entity Set: {relation_name}
Relation Description List: {relation_description_list}
#######
Output:
"""

PROMPTS[
    "summarize_relation_keywords"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given a set of entities, and a list of keywords describing the relations between the entities.
Please select some important keywords you think from the keywords list.   Make sure that these keywords summarize important events or themes of entities, including but not limited to [Main idea, major concept, or theme].  
(Note: The content of keywords should be as accurate and understandable as possible, avoiding vague or empty terms).
#######
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
#######
-Data-
Entity Set: {relation_name}
Relation Keywords List: {keywords_list}
#######
Format these keywords separated by ',' as below:
{{keyword1,keyword2,keyword3,...,keywordN}}
Output:
"""

PROMPTS[
    "entity_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entity_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""


PROMPTS["theme_keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying theme-level keywords in the user's query.

---Goal---

Given the query, extract theme-level keywords focus on overarching concepts, abstract ideas, or themes.

---Instructions---

- Focus on abstract concepts, main ideas, and overarching themes.
- Output the keywords in JSON format.
- The JSON should have one key: "theme_keywords".

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "theme_keywords": ["International trade", "Global economic stability", "Economic impact"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "theme_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "theme_keywords": ["Education", "Poverty reduction", "Socioeconomic development"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""


PROMPTS["entity_keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying entity-level keywords in the user's query.

---Goal---

Given the query, extract entity-level keywords focus on specific entities, concrete details, or specific terms.

---Instructions---

- Focus on specific entities, concrete nouns, proper names, and detailed terms.
- Output the keywords in JSON format.
- The JSON should have one key: "entity_keywords".

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "entity_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "entity_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "entity_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""


PROMPTS["naive_rag_response"] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""


PROMPTS["rag_define"] = """
Through the existing analysis, we can know that the potential keywords or theme in the query are:
{{ {entity_keywords} | {theme_keywords} }}
Please refer to keywords or theme information, combined with your own analysis, to select useful and relevant information from the prompts to help you answer accurately.
Attention: Don't brainlessly splice knowledge items! The answer needs to be as accurate, detailed, comprehensive, and convincing as possible!
"""


PROMPTS["entity_keywords_aglin"] = """---Role---

You are a helpful assistant tasked with extracting entity-level keywords from the query that are theme-aligned.

---Goal---

Given the query and theme, extract specific entities and concrete terms that relate to the identified theme.

---Instructions---

- Extract keywords that are concrete and entity-focused.
- Prioritize keywords aligned with the theme.
- Output in JSON format: {{"entity_keywords": [...]}}

######################
-Examples-
######################
Query: "What are the main causes and solutions for global poverty?"
Theme: "Poverty involves education, economic opportunity, and policy."

Output:
{{
  "entity_keywords": ["Education access", "Employment opportunities", "Economic inequality", "Social policies"]
}}
#############################

-Real Data-
######################
Query: {query}
Theme: {theme}
######################
Output:

"""


PROMPTS["rag_define_aglin"] = """
Through the existing analysis, we can know the potential theme answer and entity-level keywords are
{{ {theme_answer} | {entity_keywords} }}
Please refer to the theme answer and entity-level keywords, combined with your own analysis, to select useful and relevant information to help you answer accurately.
Attention: Don't brainlessly splice knowledge items! The answer needs to be as accurate, detailed, comprehensive, and convincing as possible!
"""



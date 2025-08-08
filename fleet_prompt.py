"""
Fleet Management Agent Prompt Template
This file contains the specialized prompt for the Fleet Management Agent.
"""

FLEET_PROMPT = """You are Bee chatbot, an EXPERT Fleet Management Agent with 20+ years of experience. You are STRICTLY LIMITED to fleet management topics only.

Your EXCLUSIVE expertise areas:
🚛 FLEET OPERATIONS: Vehicle acquisition, deployment, utilization, fleet sizing
👨‍💼 DRIVER MANAGEMENT: Recruitment, training, scheduling, performance, safety
🛣️ ROUTE OPTIMIZATION: Planning, dispatching, load optimization, delivery scheduling
🔧 MAINTENANCE & REPAIRS: Preventive maintenance, emergency repairs, parts management, service scheduling
🚗 VEHICLE DIAGNOSTICS: Engine, transmission, brake, steering, suspension, electrical system troubleshooting
🔊 SOUND-BASED DIAGNOSIS: Identifying problems through unusual noises, grinding, squealing, rattling sounds
🚙 BODY & COLLISION: Bumper, fender, paint damage, accident repairs, body alignment
⚡ PERFORMANCE ISSUES: Starting problems, overheating, stalling, check engine lights, warning indicators
⛽ FUEL MANAGEMENT: Consumption tracking, cost optimization, fuel cards, efficiency improvements
📱 FLEET TECHNOLOGY: Telematics, GPS tracking, fleet software, ELD compliance
📋 COMPLIANCE: DOT regulations, licensing, inspections, safety standards
💰 FLEET FINANCIALS: TCO, budgeting, lease vs buy, cost analysis, ROI
🛡️ RISK MANAGEMENT: Insurance, safety programs, accident management, roadside assistance
📊 ANALYTICS: KPIs, reporting, performance metrics, fleet optimization

IMPORTANT RESTRICTIONS:
❌ You CANNOT and WILL NOT answer questions about:
- General programming, coding, or software development
- Personal advice, relationships, health, or lifestyle
- Entertainment, movies, music, sports, or games
- Academic subjects unrelated to fleet management
- Weather, news, politics, or current events
- Cooking, recipes, fashion, or consumer products
- Any topic not directly related to fleet management

RESPONSE STYLE:
✅ Be direct, actionable, and results-focused
✅ Provide specific metrics, benchmarks, and best practices
✅ Include cost considerations and ROI when relevant
✅ Reference industry standards and compliance requirements
✅ Offer step-by-step implementation guidance
✅ Always consider safety and efficiency
✅ For vehicle problems: identify symptoms, possible causes, diagnostic steps, and safety considerations
✅ For sound-based issues: determine sound type, location, frequency, driving conditions when it occurs
✅ Provide step-by-step troubleshooting and when to seek professional help

If asked about non-fleet topics, you MUST respond: "I'm Bee chatbot, your specialized fleet agent, and I can only assist with fleet operations, vehicle management, driver coordination, and related logistics topics. Please ask me about fleet management instead!"

You are the go-to expert for making fleets more efficient, cost-effective, and compliant."""

# Fleet-related keywords for topic detection
FLEET_KEYWORDS = [
    # Core fleet terms
    'fleet', 'vehicle', 'car', 'truck', 'bus', 'van', 'motorcycle', 'trailer', 'semi',
    'commercial vehicle', 'company car', 'fleet vehicle', 'auto fleet', 'motor pool',
    'automobile', 'auto', 'suv', 'pickup', 'sedan', 'coupe', 'hatchback', 'wagon',
    'minivan', 'crossover', 'jeep', 'rv', 'camper', 'motorhome',
    
    # Vehicle Brands (Major Car Manufacturers)
    'toyota', 'honda', 'ford', 'chevrolet', 'chevy', 'nissan', 'hyundai', 'kia',
    'volkswagen', 'vw', 'bmw', 'mercedes', 'audi', 'lexus', 'acura', 'infiniti',
    'mazda', 'subaru', 'mitsubishi', 'volvo', 'jaguar', 'land rover', 'porsche',
    'tesla', 'cadillac', 'buick', 'gmc', 'lincoln', 'chrysler', 'dodge', 'ram',
    'jeep', 'fiat', 'alfa romeo', 'maserati', 'ferrari', 'lamborghini', 'bentley',
    'rolls royce', 'mclaren', 'aston martin', 'lotus', 'mini', 'smart', 'scion',
    'genesis', 'peugeot', 'renault', 'citroen', 'skoda', 'seat', 'opel', 'vauxhall',
    
    # Commercial Vehicle Brands
    'peterbilt', 'kenworth', 'freightliner', 'mack', 'volvo trucks', 'international',
    'western star', 'sterling', 'isuzu', 'hino', 'mitsubishi fuso', 'ud trucks',
    'ford transit', 'mercedes sprinter', 'nissan nv200', 'chevrolet express',
    'gmc savana', 'ram promaster', 'ford e-series',
    
    # Popular Vehicle Models (Common Fleet Vehicles)
    'camry', 'corolla', 'prius', 'rav4', 'highlander', 'sienna', 'tacoma', 'tundra',
    'accord', 'civic', 'cr-v', 'pilot', 'odyssey', 'ridgeline', 'fit',
    'f-150', 'f-250', 'f-350', 'explorer', 'escape', 'focus', 'fusion', 'mustang',
    'silverado', 'equinox', 'tahoe', 'suburban', 'malibu', 'impala', 'cruze',
    'altima', 'sentra', 'rogue', 'pathfinder', 'titan', 'frontier', 'leaf',
    'elantra', 'sonata', 'tucson', 'santa fe', 'genesis', 'veloster',
    'optima', 'forte', 'sorento', 'sportage', 'soul', 'rio', 'stinger',
    
    # Personnel
    'driver', 'chauffeur', 'operator', 'fleet manager', 'dispatcher', 'mechanic',
    'technician', 'fleet coordinator', 'logistics coordinator',
    
    # Operations
    'route', 'routing', 'dispatch', 'scheduling', 'delivery', 'transport', 'transportation',
    'logistics', 'cargo', 'freight', 'shipment', 'load', 'pickup', 'drop-off',
    'distribution', 'supply chain', 'last mile',
    
    # Maintenance & Service
    'maintenance', 'repair', 'service', 'inspection', 'preventive maintenance',
    'breakdown', 'downtime', 'workshop', 'garage', 'parts', 'spare parts',
    'oil change', 'tire', 'brake', 'engine', 'transmission', 'battery',
    
    # Vehicle Systems & Components
    'steering', 'suspension', 'exhaust', 'cooling system', 'radiator', 'alternator',
    'starter', 'air conditioning', 'heating', 'windshield', 'wipers', 'lights',
    'headlights', 'taillights', 'turn signals', 'mirrors', 'doors', 'windows',
    'seat', 'seatbelt', 'dashboard', 'speedometer', 'fuel gauge', 'warning lights',
    'muffler', 'catalytic converter', 'spark plugs', 'air filter', 'fuel filter',
    'power steering', 'abs', 'airbags', 'clutch', 'differential', 'axle',
    'driveshaft', 'cv joint', 'ball joint', 'tie rod', 'shock absorber', 'strut',
    'springs', 'brake pads', 'brake rotors', 'brake fluid', 'coolant', 'antifreeze',
    'thermostat', 'water pump', 'fuel pump', 'fuel injector', 'carburetor',
    'timing belt', 'serpentine belt', 'fan belt', 'hoses', 'gaskets', 'seals',
    
    # Body & Exterior Problems
    'bumper', 'front bumper', 'rear bumper', 'bumper damage', 'bumper repair',
    'body damage', 'collision repair', 'accident damage', 'dent repair',
    'paint damage', 'rust', 'scratches', 'fender', 'hood', 'trunk', 'tailgate',
    
    # Sound-Based Diagnostics
    'scraping sound', 'grinding sound', 'rattling sound', 'clicking sound',
    'squealing sound', 'squeaking sound', 'knocking sound', 'hissing sound',
    'whining sound', 'humming sound', 'buzzing sound', 'ticking sound',
    'noise when driving', 'sound from engine', 'sound from brakes',
    'sound from wheels', 'sound when turning', 'sound when accelerating',
    'sound when braking', 'sound from exhaust', 'loud noise', 'unusual noise',
    
    # Performance Issues
    'engine problems', 'wont start', 'hard to start', 'stalling', 'rough idle',
    'overheating', 'smoking', 'leaking', 'vibration', 'pulling to one side',
    'steering problems', 'brake problems', 'transmission problems',
    'acceleration problems', 'poor fuel economy', 'check engine light',
    'warning light', 'dashboard lights', 'electrical problems',
    
    # Operational Issues
    'vehicle down', 'out of service', 'towing', 'roadside assistance',
    'emergency repair', 'fleet availability', 'vehicle reliability',
    
    # Fuel & Efficiency
    'fuel', 'gas', 'diesel', 'fuel efficiency', 'mpg', 'fuel consumption',
    'fuel card', 'fuel management', 'fuel cost', 'fuel station', 'refueling',
    
    # Technology & Tracking
    'gps', 'tracking', 'telematics', 'fleet tracking', 'vehicle tracking',
    'navigation', 'route optimization', 'fleet software', 'fleet management system',
    'dashcam', 'eld', 'electronic logging device',
    
    # Compliance & Legal
    'license', 'registration', 'insurance', 'dot', 'cdl', 'hours of service',
    'compliance', 'regulation', 'safety', 'accident', 'violation',
    'fleet policy', 'dot inspection',
    
    # Financial
    'fleet cost', 'tco', 'total cost of ownership', 'lease', 'fleet lease',
    'depreciation', 'resale value', 'fleet budget', 'cost per mile',
    'operational cost', 'fleet expense',
    
    # Additional Automotive Terms
    'motor', 'automotive', 'mechanic', 'garage', 'dealership', 'service center',
    'auto repair', 'car repair', 'vehicle repair', 'mechanical', 'diagnostic',
    'tune-up', 'oil change', 'tire rotation', 'wheel alignment', 'balancing',
    'inspection', 'emissions test', 'smog check', 'registration', 'title',
    'warranty', 'recall', 'vin', 'odometer', 'mileage', 'driving',
    'parking', 'towing capacity', 'payload', 'horsepower', 'torque',
    'mpg', 'fuel economy', 'hybrid', 'electric vehicle', 'ev', 'charging',
    'manual transmission', 'automatic transmission', 'cvt', '4wd', 'awd',
    'front wheel drive', 'rear wheel drive', 'all terrain', 'off road',
    
    # Common Vehicle Issues & Symptoms
    'not starting', 'stalling', 'rough idling', 'misfiring', 'overheating',
    'leaking oil', 'burning oil', 'smoking exhaust', 'poor acceleration',
    'hard shifting', 'grinding gears', 'slipping clutch', 'brake squealing',
    'steering wheel shake', 'car pulling', 'vibration', 'noise', 'rattling',
    'squeaking', 'grinding', 'clicking', 'knocking', 'pinging', 'hissing',
    'warning light', 'check engine', 'abs light', 'oil pressure',
    'temperature gauge', 'battery light', 'charging system',
    
    # Vehicle Types & Categories
    'compact car', 'mid-size', 'full-size', 'luxury car', 'sports car',
    'convertible', 'roadster', 'muscle car', 'classic car', 'vintage',
    'economy car', 'hybrid car', 'electric car', 'plug-in hybrid',
    'work truck', 'delivery truck', 'box truck', 'dump truck', 'tow truck',
    'fire truck', 'ambulance', 'police car', 'taxi', 'uber', 'lyft',
    'rental car', 'used car', 'new car', 'certified pre-owned'
]

# Non-fleet indicators for topic filtering
NON_FLEET_INDICATORS = [
    'weather', 'recipe', 'cooking', 'health', 'medical', 'sports', 'music',
    'movie', 'entertainment', 'personal', 'relationship', 'dating',
    'school', 'education', 'homework', 'math', 'science', 'history',
    'programming', 'code', 'software', 'game', 'gaming', 'fashion',
    'beauty', 'travel vacation', 'holiday', 'restaurant', 'food',
    'politics', 'news', 'celebrity', 'joke', 'funny', 'meme'
]

# Fleet response validation indicators
FLEET_RESPONSE_INDICATORS = [
    'fleet', 'vehicle', 'driver', 'maintenance', 'fuel', 'route',
    'dispatch', 'tracking', 'logistics', 'transportation', 'truck',
    'car', 'van', 'bus', 'delivery', 'cargo', 'freight'
]

# Fleet context keywords for additional analysis
FLEET_CONTEXTS = [
    'manage', 'optimize', 'track', 'monitor', 'coordinate', 'schedule',
    'reduce cost', 'improve efficiency', 'maintain', 'operate',
    'deploy', 'utilize', 'plan', 'budget', 'analyze', 'fix', 'repair',
    'diagnose', 'troubleshoot', 'service', 'replace', 'upgrade',
    'buy', 'purchase', 'lease', 'rent', 'sell', 'trade', 'finance',
    'insure', 'register', 'inspect', 'test', 'check', 'evaluate',
    'compare', 'recommend', 'suggest', 'advice', 'help', 'assist',
    'problem with', 'issue with', 'trouble with', 'broken', 'damaged',
    'maintenance on', 'service for', 'repair of', 'parts for'
] 
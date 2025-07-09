import requests
import json
import time
import csv
import os
import pandas as pd  # Add pandas for easy CSV handling
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re

# Configuration
STEAM_API_KEY = os.getenv('STEAM_API_KEY')  # Read from environment variable
TARGET_GAMES = 3000
REQUEST_DELAY = 0.5  # Seconds between requests
MAX_DESCRIPTION_LENGTH = 2000

class SteamGameCollector:
    """Collects game data from Steam Store API"""
    
    def __init__(self, api_key: Optional[str] = None, target_games: int = TARGET_GAMES):
        self.api_key = api_key
        self.target_games = target_games
        self.collected_games = []
        
    def get_popular_game_ids(self, limit: Optional[int] = None) -> List[str]:
        """Get list of popular Steam game IDs from multiple sources"""
        if limit is None:
            limit = self.target_games * 2  # Get extra in case some fail
            
        app_ids = []
        
        # SteamSpy 'all' endpoint (paginated, 1000 games per page)
        print("Fetching games from SteamSpy 'all' endpoint...")
        try:
            pages_needed = min(10, (limit // 1000) + 1)  # Max 10 pages = 10,000 games
            
            for page in range(pages_needed):
                print(f"Fetching page {page + 1}/{pages_needed}...")
                
                url = f"https://steamspy.com/api.php?request=all&page={page}"
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    page_ids = list(data.keys())
                    app_ids.extend(page_ids)
                    print(f"Got {len(page_ids)} games from page {page + 1}")
                    
                    # Stop if we got fewer than 1000 games (last page)
                    if len(page_ids) < 1000:
                        break
                else:
                    print(f"Failed to fetch page {page + 1}: HTTP {response.status_code}")
                    break
                
                # Rate limiting for SteamSpy (1 request per minute for 'all')
                if page < pages_needed - 1:  # Don't wait after last page
                    print("Waiting 60 seconds for SteamSpy rate limit...")
                    time.sleep(60)
                    
        except Exception as e:
            print(f"SteamSpy 'all' request failed: {e}")
        
        # Remove duplicates and limit
        unique_ids = list(dict.fromkeys(app_ids))  # Preserves order
        print(f"Total unique game IDs collected: {len(unique_ids)}")
        
        return unique_ids[:limit]
    
    def fetch_game_details(self, app_id: str) -> Optional[Dict]:
        """Fetch detailed game information from Steam Store API"""
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=us&l=english"
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            if app_id not in data or not data[app_id]['success']:
                return None
            
            game_data = data[app_id]['data']
            
            # Extract basic information
            name = game_data.get('name', '')
            if not name:
                return None
            
            # Extract and clean descriptions
            detailed_desc = self._clean_description(game_data.get('detailed_description', ''))
            short_desc = self._clean_description(game_data.get('short_description', ''))
            
            # Skip games with insufficient description
            if len(detailed_desc) < 50 and len(short_desc) < 30:
                return None
            
            # Skip non-games (DLC, software, etc.)
            game_type = game_data.get('type', '').lower()
            if game_type not in ['game', '']:
                return None
            
            # Extract metadata
            genres = self._extract_genres(game_data)
            categories = self._extract_categories(game_data)
            developers = game_data.get('developers', [])
            publishers = game_data.get('publishers', [])
            
            # Extract price information
            price = self._extract_price(game_data)
            
            # Extract release date
            release_date = game_data.get('release_date', {}).get('date', '')
            
            # Extract platform support
            platforms = game_data.get('platforms', {})
            
            # Extract review information
            metacritic_score = None
            if 'metacritic' in game_data:
                metacritic_score = game_data['metacritic'].get('score')
            
            # Extract supported languages
            languages = game_data.get('supported_languages', '')
            
            return {
                'app_id': app_id,
                'name': name,
                'detailed_description': detailed_desc,
                'short_description': short_desc,
                'genres': genres,
                'categories': categories,
                'developers': developers,
                'publishers': publishers,
                'price': price,
                'release_date': release_date,
                'metacritic_score': metacritic_score,
                'platforms': platforms,
                'supported_languages': languages,
                'header_image': game_data.get('header_image', ''),
                'website': game_data.get('website', ''),
                'requirements': game_data.get('pc_requirements', {}),
                'legal_notice': game_data.get('legal_notice', ''),
                'achievements': game_data.get('achievements', {}).get('total', 0) if game_data.get('achievements') else 0
            }
            
        except Exception as e:
            print(f"Error fetching game {app_id}: {e}")
            return None
    
    def _clean_description(self, text: str) -> str:
        """Clean HTML tags and format text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove excessive special characters
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\:\;\"\'&]', '', text)
        
        # Limit length
        if len(text) > MAX_DESCRIPTION_LENGTH:
            text = text[:MAX_DESCRIPTION_LENGTH] + "..."
        
        return text
    
    def _extract_genres(self, game_data: Dict) -> List[str]:
        """Extract genre information"""
        genres = []
        if 'genres' in game_data:
            genres = [genre['description'] for genre in game_data['genres']]
        return genres
    
    def _extract_categories(self, game_data: Dict) -> List[str]:
        """Extract category/tag information"""
        categories = []
        if 'categories' in game_data:
            categories = [cat['description'] for cat in game_data['categories']]
        return categories
    
    def _extract_price(self, game_data: Dict) -> Optional[float]:
        """Extract price information"""
        if 'price_overview' in game_data:
            return game_data['price_overview'].get('final', 0) / 100.0
        elif game_data.get('is_free'):
            return 0.0
        else:
            return None
    
    def collect_games(self) -> List[Dict]:
        """Main method to collect game data"""
        print(f"Starting collection of {self.target_games} games...")
        
        # Get game IDs
        app_ids = self.get_popular_game_ids()
        
        collected_games = []
        failed_count = 0
        
        for i, app_id in enumerate(app_ids):
            if len(collected_games) >= self.target_games:
                break
            
            print(f"Fetching game {len(collected_games)+1}/{self.target_games} (ID: {app_id})", end='\r')
            
            game_data = self.fetch_game_details(app_id)
            
            if game_data:
                collected_games.append(game_data)
            else:
                failed_count += 1
            
            # Rate limiting
            time.sleep(REQUEST_DELAY)
            
            # Progress update every 50 games
            if (i + 1) % 50 == 0:
                print(f"\nProgress: {len(collected_games)} collected, {failed_count} failed")
        
        print(f"\nCollection complete: {len(collected_games)} games collected")
        self.collected_games = collected_games
        return collected_games
    
    def save_to_csv(self, filename: str = "steam_games.csv"):
        """Save collected games to CSV file using pandas (much easier)"""
        if not self.collected_games:
            print("No games to save")
            return
        
        print(f"Saving {len(self.collected_games)} games to {filename}...")
        
        # Convert to pandas DataFrame (handles all the complexity automatically)
        df = pd.DataFrame(self.collected_games)
        
        # Select only essential columns for recommendation system
        essential_columns = [
            'app_id', 'name', 'detailed_description', 'short_description',
            'genres', 'categories', 'price', 'release_date', 'developers', 'publishers'
        ]
        
        # Keep only columns that exist in the data
        available_columns = [col for col in essential_columns if col in df.columns]
        df_filtered = df[available_columns]
        
        # Convert lists to strings for CSV compatibility
        for col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x) if x is not None else ''
            )
        
        # Save to CSV
        df_filtered.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"Data saved to {filename}")
        print(f"Columns saved: {list(df_filtered.columns)}")
    
def main():
    """Main execution function"""
    print("Steam Game Data Collector")
    print("=" * 40)
    
    # Initialize collector
    collector = SteamGameCollector(target_games=TARGET_GAMES)
    
    # Collect new data
    games = collector.collect_games()
    
    if games:
        # Save data
        collector.save_to_csv()  # Uses pandas - easiest method
        
        # Display summary
        print(f"\nCollection Summary:")
        print(f"Total games: {len(games)}")
        print(f"Average description length: {sum(len(g['detailed_description']) for g in games) / len(games):.0f} chars")
        
        # Genre distribution
        all_genres = []
        for game in games:
            all_genres.extend(game['genres'])
        
        from collections import Counter
        genre_counts = Counter(all_genres)
        print(f"Top genres: {list(genre_counts.most_common(5))}")
        
    return collector

if __name__ == "__main__":
    collector = main()
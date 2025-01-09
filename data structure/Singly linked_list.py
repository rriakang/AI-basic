class SongNode:
    def __init__(self, title, artist, duration):
        self.title = title
        self.artist = artist
        self.duration = duration
        self.next = None

class Playlist:
    def __init__(self):
        self.head = None

    def add_song(self, title, artist, duration):
        new_song = SongNode(title, artist, duration)
        if not self.head:
            self.head = new_song
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_song

    def delete_song(self, title):
        if not self.head:
            print("재생 목록이 비어 있습니다.")
            return

        # 첫 번째 곡 삭제
        if self.head.title == title:
            self.head = self.head.next
            print(f"{title}이(가) 재생 목록에서 삭제되었습니다.")
            return

        # 중간 또는 마지막 곡 삭제
        current = self.head
        while current.next and current.next.title != title:
            current = current.next

        if current.next:
            current.next = current.next.next
            print(f"{title}이(가) 재생 목록에서 삭제되었습니다.")
        else:
            print(f"{title}을(를) 찾을 수 없습니다.")

    def play(self):
        current = self.head
        while current:
            print(f"재생 중: {current.title} - {current.artist} ({current.duration}분)")
            current = current.next

# 예제 사용
playlist = Playlist()
playlist.add_song("Song A", "Artist A", 3)
playlist.add_song("Song B", "Artist B", 4)
playlist.add_song("Song C", "Artist C", 5)

print("재생 목록:")
playlist.play()

print("\n곡 삭제:")
playlist.delete_song("Song B")
playlist.play()
 